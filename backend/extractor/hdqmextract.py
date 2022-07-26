#!/usr/bin/env python
from __future__ import print_function

from glob import glob
from multiprocessing import Pool
from collections import defaultdict
from configparser import RawConfigParser
from tempfile import NamedTemporaryFile
from sqlalchemy.exc import IntegrityError

import re
import ROOT
import errno
import argparse

import metrics
from metrics import fits
from metrics import basic

import os, sys
# Insert parent dir to sys.path to import db_access
sys.path.insert(1, os.path.realpath(os.path.pardir))
import db_access
from helpers import batch_iterable, exec_transaction, get_all_me_names

CFGFILES = 'cfg/*/*.ini'
ROOTFILES = '/eos/cms/store/group/comm_dqm/DQMGUI_data/*/*/*/DQM*.root'

PDPATTERN = re.compile('DQM_V\d+_R\d+__(.+__.+__.+)[.]root') # PD inside the file name
VERSIONPATTERN = re.compile('(DQM_V)(\d+)(.+[.]root)')
RUNPATTERN = re.compile('DQM_V\d+_R0+(\d+)__.+[.]root')
PLOTNAMEPATTERN = re.compile('^[a-zA-Z0-9_+-]*$')
DQMGUI = 'https://cmsweb.cern.ch/dqm/offline/'


def get_full_path(relativePath, run):
  parts = relativePath.split('/')
  return str('DQMData/Run %s/%s/Run summary/%s' % (run, parts[0], '/'.join(parts[1:])))


def remove_old_versions(all_files):
  # groups is a map: filename with version part removed -> list of all files 
  # with the same name but different version.
  groups = defaultdict(list)
  for fullpath in all_files:
    filename = fullpath.split('/')[-1]
    version = 1
    mapKey = filename
    versionMatch = VERSIONPATTERN.findall(filename)
    # We should get 3 parts: DQM_V, version and the rest of the file name
    if len(versionMatch) == 1 and len(versionMatch[0]) == 3:
      version = int(versionMatch[0][1])
      # Key is everything appart from version
      mapKey = versionMatch[0][0] + versionMatch[0][2]
  
    obj = {}
    obj['fullpath'] = fullpath
    obj['filename'] = filename
    obj['version'] = version
    
    groups[mapKey].append(obj)

  # Sort every group by version and select the latest one
  files = map(lambda x: sorted(groups[x], key=lambda elem: elem['version'], reverse=True)[0]['fullpath'], groups)

  return files


# Write an me to a tempfile and read binary from it.
# This is to keep the compatibility with future ROOT versions.
def get_binary(me):
  with NamedTemporaryFile(dir='/dev/shm/') as temp_file:
    result_file = ROOT.TFile(temp_file.name, 'recreate')
    me.Write()
    result_file.Close()
    with open(temp_file.name, "rb") as file:
      return file.read()


def get_all_available_runs():
  eos_files = glob(ROOTFILES)
  runs = set()
  for filepath in eos_files:
    file = filepath.split('/')[-1]
    run_match = RUNPATTERN.findall(file)
    if not len(run_match) == 0:
      run = run_match[0]
      runs.add(int(run))
  return list(runs)


def extract_all_mes(cfg_files, runs, nprocs, all_files, in_dataset, in_query, from_run, to_run):
  print('Processing %d configuration files...' % len(cfg_files))
  mes_set = set()
  good_files = 0
  for cfg_file in cfg_files:
    try:
      parser = RawConfigParser()
      parser.read(unicode(cfg_file))
      for section in parser:
        if in_query and not in_query in section:
          continue
        if not section.startswith('plot:'):
          if section != 'DEFAULT':
            print('Invalid configuration section: %s:%s, skipping.' % (cfg_file, section))
          continue
        if not PLOTNAMEPATTERN.match(section.lstrip('plot:')):
          print("Invalid plot name: '%s:%s' Plot names can contain only: [a-zA-Z0-9_+-]" % (cfg_file, section.lstrip('plot:')))
          continue

        mes_set.update(get_all_me_names(parser[section]['relativePath']))
        if 'histo1Path' in parser[section]:
          mes_set.update(get_all_me_names(parser[section]['histo1Path']))
        if 'histo2Path' in parser[section]:
          mes_set.update(get_all_me_names(parser[section]['histo2Path']))
        if 'reference' in parser[section]:
          mes_set.update(get_all_me_names(parser[section]['reference']))
      good_files += 1
    except:
      print('Could not read %s, skipping...' % cfg_file)

  print('Read %d configuration files.' % good_files)
  print('Read %d distinct ME paths.' % len(mes_set))

  if not all_files:
    print('Listing files on EOS, this can take a while...')
    eos_string = ROOTFILES
    if in_dataset:
      eos_string = eos_string.replace('DQMGUI_data/*/*', 'DQMGUI_data/*/%s' % in_dataset)
    print(eos_string)
    all_files = glob(eos_string)
    if len(all_files) == 0:
      print('GLOB returned 0 files, probably EOS is down. Aborting.')
      return
    print('Done.')
  else:
    print('Using provided DQM files: %s' % len(all_files))

  # Filter on the runs that were passed by the user
  if runs or (from_run and to_run):
    filtered = []
    for file in all_files:
      run_match = RUNPATTERN.findall(file)
      if not len(run_match) == 0:
        run = run_match[0]
        if (runs and int(run) in runs) or \
           ((from_run and to_run) and \
            (int(run) >= from_run and \
             int(run) <= to_run)):
          filtered.append(file)
    all_files = filtered

  # Keep only the newest version of each file
  print('Removing old versions of files...')
  all_files = remove_old_versions(all_files)

  print('Found %s files in EOS' % len(all_files))

  print('Gathering information about MEs to be extracted...')

  db_access.setup_db()

  # Get lists of existing mes and eos files.
  # Existing means that ME was extracted or is in the extraction queue.
  session = db_access.get_session()
  existing_me_paths = set(x.me_path for x in session.query(db_access.TrackedMEPathForMEExtraction).all())
  existing_eos_paths = set(x.eos_path for x in session.query(db_access.TrackedEOSPathForMEExtraction).all())
  session.close()

  # Single session (transaction) for queue manipulations
  session = db_access.get_session()

  # -------------------- Update the ME paths in the extraction queue -------------------- #
  new_mes = mes_set.difference(existing_me_paths)
  deleted_mes = existing_me_paths.difference(mes_set)

  print('New MEs: %s, deleted MEs: %s' % (len(new_mes), len(deleted_mes)))

  # Remove deleted MEs from the extraction queue
  if len(deleted_mes) > 0:
    sql = 'DELETE FROM queue_to_extract WHERE me_path = :me_path;'
    session.execute(sql, [{'me_path': x} for x in deleted_mes])

    sql = 'DELETE FROM tracked_me_paths_for_me_extraction WHERE me_path = :me_path;'
    session.execute(sql, [{'me_path': x} for x in deleted_mes])

  # Refresh new MEs table
  sql = 'DELETE FROM new_me_paths_for_me_extraction;'
  session.execute(sql)

  # Insert new ME paths
  if len(new_mes) > 0:
    sql = 'INSERT INTO new_me_paths_for_me_extraction (me_path) VALUES (:me_path);'
    session.execute(sql, [{'me_path': x} for x in new_mes])

  # Will have to extract new MEs for every existing file
  sql_update_queue = '''
  INSERT INTO queue_to_extract (eos_path, me_path)
  SELECT eos_path, me_path
  FROM tracked_eos_paths_for_me_extraction, new_me_paths_for_me_extraction
  ;
  '''

  sql_update_existing = '''
  INSERT INTO tracked_me_paths_for_me_extraction (me_path)
  SELECT me_path
  FROM new_me_paths_for_me_extraction
  ;
  '''

  session.execute(sql_update_queue)
  session.execute(sql_update_existing)

  # -------------------- Update the eos paths in the extraction queue -------------------- #
  files_set = set(all_files)
  new_files = files_set.difference(existing_eos_paths)
  deleted_files = existing_eos_paths.difference(files_set)

  print('New files: %s, deleted files: %s' % (len(new_files), len(deleted_files)))

  # Remove deleted files from the extraction queue
  if len(deleted_files) > 0:
    sql = 'DELETE FROM queue_to_extract WHERE eos_path = :eos_path;'
    session.execute(sql, [{'eos_path': x} for x in deleted_files])

    sql = 'DELETE FROM tracked_eos_paths_for_me_extraction WHERE eos_path = :eos_path;'
    session.execute(sql, [{'eos_path': x} for x in deleted_files])

  # Refresh new files table
  sql = 'DELETE FROM new_eos_paths_for_me_extraction;'
  session.execute(sql)

  # Insert new eos paths
  if len(new_files) > 0:
    sql = 'INSERT INTO new_eos_paths_for_me_extraction (eos_path) VALUES (:eos_path);'
    session.execute(sql, [{'eos_path': x} for x in new_files])

  # Will have to extract all existing MEs for newly added files
  sql_update_queue = '''
  INSERT INTO queue_to_extract (eos_path, me_path)
  SELECT eos_path, me_path
  FROM new_eos_paths_for_me_extraction, tracked_me_paths_for_me_extraction
  ;
  '''

  sql_update_existing = '''
  INSERT INTO tracked_eos_paths_for_me_extraction (eos_path)
  SELECT eos_path
  FROM new_eos_paths_for_me_extraction
  ;
  '''

  session.execute(sql_update_queue)
  session.execute(sql_update_existing)

  session.commit()
  session.close()

  print('Done.')
  print('Extracting missing MEs...')

  # ------------------- Start extracting MEs from the extraction queue ------------------- #

  sql = 'SELECT id, eos_path, me_path FROM queue_to_extract LIMIT 100000;'
  pool = Pool(nprocs)

  while True:
    db_access.dispose_engine()
    session = db_access.get_session()
    try:
      print('Fetching not processed MEs from DB...')
      rows = session.execute(sql)
      rows = list(rows)
      session.close()
      print('Fetched: %s' % len(rows))
      if len(rows) == 0:
        break

      pool.map(extract_mes, batch_iterable(rows, chunksize=2000))
    except OSError as e:
      if e.errno != errno.EINTR:
        raise
      else:
        print('[Errno 4] occurred. Continueing.')
    except Exception as e:
      print(e)
      session.close()

  print('Done.')


def extract_mes(rows):
  db_access.dispose_engine()
  tdirectory = None
  last_eos_path = None

  nRows = len(rows)
  iRow = 0
  for row in rows:
    iRow += 1
    id = row['id']
    eos_path = row['eos_path']
    me_path = row['me_path']

    pd_match = PDPATTERN.findall(eos_path)
    if len(pd_match) == 0:
      dataset = ''
    else:
      dataset = '/' + pd_match[0].replace('__', '/')

    filename = eos_path.split('/')[-1]
    run_match = RUNPATTERN.findall(filename)
    if not len(run_match) == 0:
      run = run_match[0]
    else:
      print('Skipping a malformatted DQM file that does not contain a run number: %s' % eos_path)
      exec_transaction('DELETE FROM queue_to_extract WHERE id = :id', {'id': id})
      continue

    # Open root file only if it's different from the last one
    if eos_path != last_eos_path or tdirectory == None:
      if tdirectory != None:
        tdirectory.Close()
      tdirectory = ROOT.TFile.Open(eos_path)
      last_eos_path = eos_path

    if tdirectory == None:
      print("Unable to open file: '%s'" % eos_path)
      exec_transaction('DELETE FROM queue_to_extract WHERE id = :id', {'id': id})
      continue

    fullpath = get_full_path(me_path, run)
    plot = tdirectory.Get(fullpath)
    if not plot:
      exec_transaction('DELETE FROM queue_to_extract WHERE id = :id', {'id': id})
      continue

    plot_folder = '/'.join(me_path.split('/')[:-1])
    gui_url = '%sstart?runnr=%s;dataset=%s;workspace=Everything;root=%s;focus=%s;zoom=yes;' % (DQMGUI, run, dataset, plot_folder, me_path)
    image_url = '%splotfairy/archive/%s%s/%s?v=1510330581101995531;w=1906;h=933' % (DQMGUI, run, dataset, me_path)
    monitor_element = db_access.MonitorElement(
      run = run,
      lumi = 0,
      eos_path = eos_path,
      me_path = me_path,
      dataset = dataset,
      gui_url = gui_url,
      image_url = image_url)

    session = db_access.get_session()
    try:
      me_blob = db_access.MeBlob(
        me_blob = get_binary(plot)
      )
      session.add(me_blob)
      session.flush()
      monitor_element.me_blob_id = me_blob.id
      session.add(monitor_element)
      session.flush()
      session.execute('DELETE FROM queue_to_extract WHERE id = :id;', {'id': id})
      session.execute('INSERT INTO queue_to_calculate (me_id) VALUES (:me_id);', {'me_id': monitor_element.id})
      session.commit()
      if iRow == 1 or iRow % int(nRows / 10) == 0:
        print('Added ME %s (#%d / %d) to DB: %s:%s' % (monitor_element.id, iRow, nRows, eos_path, me_path))
    except IntegrityError as e:
      print('Insert ME IntegrityError: %s' % e)
      # ME already exists. Remove it from the queue_to_extract and add to queue_to_calculate
      # because it is possible that it wasn't calculated yet
      session.rollback()
      monitor_element_id = monitor_element.id
      session.execute('DELETE FROM queue_to_extract WHERE id = :id;', {'id': id})
      if monitor_element_id == None:
        res = session.execute('SELECT id FROM monitor_elements WHERE eos_path=:eos_path AND me_path=:me_path;', 
          {'eos_path': eos_path, 'me_path': me_path})
        monitor_element_id = list(res)[0][0]
      session.execute('INSERT INTO queue_to_calculate (me_id) VALUES (:me_id);', {'me_id': monitor_element_id})
      session.commit()
    except Exception as e:
      print('Insert ME error: %s' % e)
      session.rollback()
    finally:
      session.close()

  if tdirectory:
    tdirectory.Close()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='HDQM trend calculation.')
  parser.add_argument('-r', dest='runs', type=int, nargs='+', help='Runs to process. If none were given, will process all available runs.')
  parser.add_argument('-c', dest='config', nargs='+', help='Configuration files to process. If none were given, will process all available configuration files. Files must come from here: cfg/*/*.ini')
  parser.add_argument('-j', dest='nprocs', type=int, default=50, help='Number of processes to use for extraction.')
  parser.add_argument('-f', dest='files', type=str, nargs='+', help='DQM TDirectory ROOT files to take MEs from. If not provided, a dedicated EOS directory will be used.')
  parser.add_argument('--dataset', dest='in_dataset', type=str, default=None, help='Primary datset to process. If none given, will process all available datatsets.')
  parser.add_argument('--query', dest='in_query', type=str, default=None, help='Trend metric query. If none given, will process all available trends.')
  parser.add_argument('--from_run', dest='from_run', type=int, default=None, help='First run to process')
  parser.add_argument('--to_run', dest='to_run', type=int, default=None, help='Last run to process')
  args = parser.parse_args()

  runs = args.runs
  config = args.config
  nprocs = args.nprocs
  all_files = args.files

  if nprocs < 0:
    print('Number of processes must be a positive integer')
    exit()

  if config == None:
    config = glob(CFGFILES)

  # Validate config files
  for cfg_file in config:
    if cfg_file.count('/') != 2 or not cfg_file.startswith('cfg/'):
      print('Invalid configuration file: %s' % cfg_file)
      print('Configuration files must come from here: cfg/*/*.ini')
      exit()

  extract_all_mes(config, runs, nprocs, all_files, \
                  args.in_dataset, args.in_query, \
                  args.from_run, args.to_run)
