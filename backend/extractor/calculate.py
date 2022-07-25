#!/usr/bin/env python
from __future__ import print_function

from sys import argv
from glob import glob
from multiprocessing import Pool
from collections import defaultdict
from configparser import RawConfigParser
from tempfile import NamedTemporaryFile
from sqlalchemy.exc import IntegrityError

import re
import math
import ROOT
import errno
import tempfile
import argparse

from metrics import fits
from metrics import basic
from metrics import muon_metrics
from metrics import L1T_metrics
from metrics import hcal_metrics

import os, sys
# Insert parent dir to sys.path to import db_access
sys.path.insert(1, os.path.realpath(os.path.pardir))
import db_access
from ForkPool import ForkPool
from helpers import batch_iterable, exec_transaction, get_all_me_names

METRICS_MAP = {'fits': fits, 'basic': basic, 'L1T_metrics': L1T_metrics, 'muon_metrics': muon_metrics, 'hcal_metrics': hcal_metrics}

PLOTNAMEPATTERN = re.compile('^[a-zA-Z0-9_+-]*$')
CFGFILES = 'cfg/*/*.ini'

# Globallist of configs will be shared between processes
CONFIG=[]


def get_optional_me(eos_path, me_paths):
  sql = 'SELECT monitor_elements.id, monitor_elements.gui_url, monitor_elements.image_url, me_blob FROM monitor_elements JOIN me_blobs ON monitor_elements.me_blob_id= me_blobs.id WHERE eos_path=:eos_path AND me_path=:me_path;'
  for me_path in me_paths:
    session = db_access.get_session()
    me=None
    try:
      me = session.execute(sql, {'eos_path': eos_path, 'me_path': me_path})
      me = list(me)
    except Exception as e:
      print(e)
    finally:
      session.close()

    if me:
      return me[0]['id'], me[0]['me_blob'], me[0]['gui_url'], me[0]['image_url']

  # None were found
  return None, None, None, None


def get_me_blob_by_me_id(id):
  sql = 'SELECT monitor_elements.gui_url, monitor_elements.image_url, me_blob FROM monitor_elements JOIN me_blobs ON monitor_elements.me_blob_id= me_blobs.id WHERE monitor_elements.id=:id;'
  session = db_access.get_session()
  me=None
  try:
    me = session.execute(sql, {'id': id})
    me = list(me)
  except Exception as e:
    print(e)
  finally:
    session.close()

  if not me:
    return None, None, None
  return me[0]['me_blob'], me[0]['gui_url'], me[0]['image_url']


def move_to_second_queue(me_id, queue_id):
  session = db_access.get_session()
  try:
    session.execute('INSERT INTO queue_to_calculate_later (me_id) VALUES (:me_id);', {'me_id': me_id})
    session.execute('DELETE FROM queue_to_calculate WHERE id = :queue_id;', {'queue_id': queue_id})
    session.commit()
  except Exception as e:
    session.rollback()
    print(e)
  finally:
    session.close()


# Returns a ROOT plot from binary file
def get_plot_from_blob(me_blob):
  with tempfile.NamedTemporaryFile(dir='/dev/shm/') as temp_file:
    with open(temp_file.name, 'w+b') as fd:
      fd.write(me_blob)
    tdirectory = ROOT.TFile(temp_file.name, 'read')
    plot = tdirectory.GetListOfKeys()[0].ReadObj()
    return plot, tdirectory


def section_to_config_object(section):
  return db_access.LastCalculatedConfig(
    subsystem = section['subsystem'],
    name = section['name'],
    metric = section['metric'],
    plot_title = section.get('plotTitle') or section['name'],
    y_title = section['yTitle'],
    relative_path = section['relativePath'],
    histo1_path = section.get('histo1Path'),
    histo2_path = section.get('histo2Path'),
    reference_path = section.get('reference'),
    threshold = section.get('threshold'),
  )


def get_processing_string(dataset):
  try:
    return dataset.split('-')[1]
  except:
    return 'unknown'


def calculate_all_trends(cfg_files, runs, nprocs, in_dataset, in_query):
  print('Processing %d configuration files...' % len(cfg_files))
  db_access.setup_db()

  ## Paths to relevant histograms
  relative_paths = []
  
  trend_count=0
  for cfg_file in cfg_files:
    subsystem = os.path.basename(os.path.dirname(cfg_file))
    if not subsystem:
      subsystem = 'Unknown'

    parser = RawConfigParser()
    parser.read(unicode(cfg_file))

    for section in parser:
      if in_query and not in_query in section:
        continue

      if not section.startswith('plot:'):
        if(section != 'DEFAULT'):
          print('Invalid configuration section: %s:%s, skipping.' % (cfg_file, section))
        continue

      if not PLOTNAMEPATTERN.match(section.lstrip('plot:')):
        print("Invalid plot name: '%s:%s' Plot names can contain only: [a-zA-Z0-9_+-]" % (cfg_file, section.lstrip('plot:')))
        continue

      if 'metric' not in parser[section] or\
         'relativePath' not in parser[section] or\
         'yTitle' not in parser[section]:
        print('Plot missing required attributes: %s:%s, skipping.' % (cfg_file, section))
        print('Required parameters: metric, relativePath, yTitle')
        continue
      
      parser[section]['subsystem'] = subsystem
      parser[section]['name'] = section.split(':')[1]
      CONFIG.append(parser[section])
      relative_paths.append(parser[section]['relativePath'])
      trend_count+=1

  print('Starting to process %s trends.' % trend_count)
  print('Updating configuration...')

  # Find out new and changed configuration
  last_config=[]
  session = db_access.get_session()
  try:
    last_config = list(session.execute('SELECT * FROM last_calculated_configs;'))
  except Exception as e:
    print('Exception getting config from the DB: %s' % e)
    return
  finally:
    session.close()

  new_configs=[]

  for current in CONFIG:
    # Find by subsystem and name of trend
    last = next((x for x in last_config if current['subsystem'] == x['subsystem'] and current['name'] == x['name']), None)
    if last:
      obj = section_to_config_object(current)
      if not last['metric'] == obj.metric or\
        not last['plot_title'] == obj.plot_title or\
        not last['y_title'] == obj.y_title or\
        not last['relative_path'] == obj.relative_path or\
        not last['histo1_path'] == obj.histo1_path or\
        not last['histo2_path'] == obj.histo2_path or\
        not last['reference_path'] == obj.reference_path or\
        not last['threshold'] == int(obj.threshold) if obj.threshold else None:
        # Changed!
        new_configs.append(obj)
    else:
      new_configs.append(section_to_config_object(current))

  # Add new configs
  session = db_access.get_session()
  try:
    for new in new_configs:
      session.add(new)
    session.commit()
  except Exception as e:
    print('Exception adding new configs to the DB: %s' % e)
    session.rollback()
    return
  finally:
    session.close()

  # Recalculate everything if the configuration changed
  if len(new_configs) > 0:
    print('Configuration changed, reseting the calculation queue...')
    session = db_access.get_session()
    try:
      session.execute('DELETE FROM queue_to_calculate;')
      session.execute('DELETE FROM queue_to_calculate_later;')
      session.execute('INSERT INTO queue_to_calculate (me_id) SELECT id FROM monitor_elements;')
      session.commit()
    except Exception as e:
      print('Exception reseting the calculation queue in the DB: %s' % e)
      session.rollback()
      return
    finally:
      session.close()
    print('Calculation queue is ready.')
  else:
    # Move things from queue_to_calculate_later back to queue_to_calculate
    print('Moving items from second queue to the main one...')
    session = db_access.get_session()
    try:
      session.execute('INSERT INTO queue_to_calculate (me_id) SELECT me_id FROM queue_to_calculate_later;')
      session.execute('DELETE FROM queue_to_calculate_later;')
      session.commit()
    except Exception as e:
      print('Exception moving items from the second calculation queue to the first: %s' % e)
      session.rollback()
    finally:
      session.close()
    print('Calculation queue is ready.')

  print('Configuration updated.')

  # Start calculating trends
  if runs == None:
    runs_filter = ''
  else:
    runs_filter = 'WHERE monitor_elements.run IN (%s)' % ', '.join(str(x) for x in runs)

  limit = 10000
  sql = '''
  SELECT queue_to_calculate.id, monitor_elements.id as me_id, monitor_elements.run, monitor_elements.lumi, monitor_elements.eos_path, monitor_elements.me_path, monitor_elements.dataset FROM monitor_elements
  JOIN queue_to_calculate ON monitor_elements.id=queue_to_calculate.me_id
  %s
  LIMIT %s;
  ''' % (runs_filter, limit)

  # pool = Pool(nprocs)
  pool = ForkPool(nprocs)
  
  while True:
    db_access.dispose_engine()
    session = db_access.get_session()

    try:
      print('Fetching not processed data points from DB...')
      rows = session.execute(sql)
      rows = list(rows)
      if in_dataset:
        rows = list([r for r in rows if r[6].split('/')[1] == in_dataset])
      if in_query:
        rows = list([r for r in rows if r[5] in relative_paths])
      print('Fetched: %s' % len(rows))
      if len(rows) == 0:
        print('Queue to calculate is empty. Exiting.')
        break

      pool.map(calculate_trends, batch_iterable(rows, chunksize=400))
      
      print('Finished calculating a batch of trends.')
    except OSError as e:
      if e.errno != errno.EINTR:
        raise
      else:
        print('[Errno 4] occurred. Continueing.')
    except Exception as e:
      print('Exception fetching elements from the calculation queue: %s' % e)
      raise
    finally:
      session.close()


def calculate_trends(rows):
  db_access.dispose_engine()

  for row in rows:
    print('Calculating trend:', row['eos_path'], row['me_path'])
    # All configs referencing row['me_path'] as main me
    configs = [x for x in CONFIG if row['me_path'] in get_all_me_names(x['relativePath'])]
    
    if not configs:
      print('ME not used is any config')

      # Remove ME from queue_to_calculate 
      session = db_access.get_session()
      try:
        session.execute('DELETE FROM queue_to_calculate WHERE id = :id;', {'id': row['id']})
        session.commit()
      except Exception as e:
        session.rollback()
        print(e)
      finally:
        session.close()

      continue

    for config in configs:
      tdirectories=[]

      try:
        try:
          metric = eval(config['metric'], METRICS_MAP)
        except Exception as e:
          print('Unable to load the metric: %s. %s' % (config['metric'], e))
          move_to_second_queue(row['me_id'], row['id'])
          break
        
        histo1_id=None
        histo2_id=None
        reference_id=None

        histo1_gui_url=None
        histo2_gui_url=None
        reference_gui_url=None

        histo1_image_url=None
        histo2_image_url=None
        reference_image_url=None

        if 'histo1Path' in config:
          histo1_id, histo1, histo1_gui_url, histo1_image_url = get_optional_me(row['eos_path'], get_all_me_names(config['histo1Path']))
          if not histo1:
            print('Unable to get an optional monitor element 1: %s:%s' % (row['eos_path'], config['histo1Path']))
            move_to_second_queue(row['me_id'], row['id'])
            break
          plot, tdir = get_plot_from_blob(histo1)
          tdirectories.append(tdir)
          metric.setOptionalHisto1(plot)

        if 'histo2Path' in config:
          histo2_id, histo2, histo2_gui_url, histo2_image_url = get_optional_me(row['eos_path'], get_all_me_names(config['histo2Path']))
          if not histo2:
            print('Unable to get an optional monitor element 2: %s:%s' % (row['eos_path'], config['histo2Path']))
            move_to_second_queue(row['me_id'], row['id'])
            break
          plot, tdir = get_plot_from_blob(histo2)
          tdirectories.append(tdir)
          metric.setOptionalHisto2(plot)

        if 'reference' in config:
          reference_id, reference, reference_gui_url, reference_image_url = get_optional_me(row['eos_path'], get_all_me_names(config['reference']))
          if not reference:
            print('Unable to get an optional reference monitor element: %s:%s' % (row['eos_path'], config['reference']))
            move_to_second_queue(row['me_id'], row['id'])
            break
          plot, tdir = get_plot_from_blob(reference)
          tdirectories.append(tdir)
          metric.setReference(plot)

        if 'threshold' in config:
          metric.setThreshold(config['threshold'])

        # Get main plot blob from db
        main_me_blob, main_gui_url, main_image_url = get_me_blob_by_me_id(row['me_id'])

        if not main_me_blob:
          print('Unable to get me_blob %s from the DB.' % row['me_id'])
          move_to_second_queue(row['me_id'], row['id'])
          break

        main_plot, tdir = get_plot_from_blob(main_me_blob)
        tdirectories.append(tdir)

        # Get config id
        session = db_access.get_session()
        config_id=0
        try:
          config_id = session.execute('SELECT id FROM last_calculated_configs WHERE subsystem=:subsystem AND name=:name;', {'subsystem': config['subsystem'], 'name': config['name']})
          config_id = list(config_id)
          config_id = config_id[0]['id']
        except Exception as e:
          print('Unable to get config id from the DB: %s' % e)
          move_to_second_queue(row['me_id'], row['id'])
          break
        finally:
          session.close()

        # Calculate
        try:
          value, error = metric.calculate(main_plot)
        except Exception as e:
          print('Unable to calculate the metric: %s. %s' % (config['metric'], e))
          move_to_second_queue(row['me_id'], row['id'])
          break

        # Write results to the DB
        historic_data_point = db_access.HistoricDataPoint(
          run = row['run'],
          lumi = row['lumi'],
          dataset = row['dataset'],
          subsystem = config['subsystem'],
          pd = row['dataset'].split('/')[1],
          processing_string = get_processing_string(row['dataset']),
          value = value,
          error = error,
          main_me_id = row['me_id'],
          optional_me1_id = histo1_id,
          optional_me2_id = histo2_id,
          reference_me_id = reference_id,
          config_id = config_id,

          name = config['name'],
          plot_title = config.get('plotTitle') or config['name'],
          y_title = config['yTitle'],
          main_me_path = config['relativePath'],
          optional1_me_path = config.get('histo1Path'),
          optional2_me_path = config.get('histo2Path'),
          reference_path = config.get('reference'),
          main_gui_url = main_gui_url,
          main_image_url = main_image_url,
          optional1_gui_url = histo1_gui_url,
          optional1_image_url = histo1_image_url,
          optional2_gui_url = histo2_gui_url,
          optional2_image_url = histo2_image_url,
          reference_gui_url = reference_gui_url,
          reference_image_url = reference_image_url,
        )

        session = db_access.get_session()
        try:
          session.add(historic_data_point)
          session.execute('DELETE FROM queue_to_calculate WHERE id=:id;', {'id': row['id']})
          session.execute(db_access.insert_or_ignore_crossdb('INSERT INTO selection_params (subsystem, pd, processing_string, config_id) VALUES (:subsystem, :pd, :ps, :config_id);'), 
            {'subsystem': config['subsystem'], 'pd': historic_data_point.pd, 'ps': historic_data_point.processing_string, 'config_id': config_id}
          )
          session.commit()
        except IntegrityError as e:
          print('Insert HistoricDataPoint error: %s' % e)
          session.rollback()
          print('Updating...')
          try:
            historic_data_point_existing = session.query(db_access.HistoricDataPoint).filter(
              db_access.HistoricDataPoint.config_id == historic_data_point.config_id,
              db_access.HistoricDataPoint.main_me_id == historic_data_point.main_me_id,
            ).one_or_none()

            if historic_data_point_existing:
              historic_data_point_existing.run = historic_data_point.run
              historic_data_point_existing.lumi = historic_data_point.lumi
              historic_data_point_existing.dataset = historic_data_point.dataset
              historic_data_point_existing.subsystem = historic_data_point.subsystem
              historic_data_point_existing.pd = historic_data_point.pd
              historic_data_point_existing.processing_string = historic_data_point.processing_string
              historic_data_point_existing.value = historic_data_point.value
              historic_data_point_existing.error = historic_data_point.error
              historic_data_point_existing.optional_me1_id = historic_data_point.optional_me1_id
              historic_data_point_existing.optional_me2_id = historic_data_point.optional_me2_id
              historic_data_point_existing.reference_me_id = historic_data_point.reference_me_id

              historic_data_point_existing.name = historic_data_point.name
              historic_data_point_existing.plot_title = historic_data_point.plot_title
              historic_data_point_existing.y_title = historic_data_point.y_title
              historic_data_point_existing.main_me_path = historic_data_point.main_me_path
              historic_data_point_existing.optional1_me_path = historic_data_point.optional1_me_path
              historic_data_point_existing.optional2_me_path = historic_data_point.optional2_me_path
              historic_data_point_existing.reference_path = historic_data_point.reference_path
              historic_data_point_existing.main_gui_url = historic_data_point.main_gui_url
              historic_data_point_existing.main_image_url = historic_data_point.main_image_url
              historic_data_point_existing.optional1_gui_url = historic_data_point.optional1_gui_url
              historic_data_point_existing.optional1_image_url = historic_data_point.optional1_image_url
              historic_data_point_existing.optional2_gui_url = historic_data_point.optional2_gui_url
              historic_data_point_existing.optional2_image_url = historic_data_point.optional2_image_url
              historic_data_point_existing.reference_gui_url = historic_data_point.reference_gui_url
              historic_data_point_existing.reference_image_url = historic_data_point.reference_image_url

              session.execute('DELETE FROM queue_to_calculate WHERE id=:id;', {'id': row['id']})
              session.commit()
              print('Updated.')
          except Exception as e:
            print('Update HistoricDataPoint error: %s' % e)
            session.rollback()
            move_to_second_queue(row['me_id'], row['id'])
        finally:
          session.close()
      except Exception as e:
        print('Exception calculating trend: %s' % e)
        move_to_second_queue(row['me_id'], row['id'])
        break
      finally:
        # Close all open TDirectories
        for tdirectory in tdirectories:
          if tdirectory:
            tdirectory.Close()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='HDQM trend calculation.')
  parser.add_argument('-r', dest='runs', type=int, nargs='+', help='Runs to process. If none were given, will process all available runs.')
  parser.add_argument('-c', dest='config', nargs='+', help='Configuration files to process. If none were given, will process all available configuration files. Files must come from here: cfg/*/*.ini')
  parser.add_argument('-j', dest='nprocs', type=int, default=25, help='Number of processes to use for extraction.')
  parser.add_argument('--dataset', dest='in_dataset', type=str, default=None, help='Primary datset to process. If none given, will process all available datatsets.')
  parser.add_argument('--query', dest='in_query', type=str, default=None, help='Trend metric query. If none given, will process all available trends.')
  args = parser.parse_args()

  runs = args.runs
  config = args.config
  nprocs = args.nprocs
  in_dataset = args.in_dataset
  in_query = args.in_query

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

  calculate_all_trends(config, runs, nprocs, in_dataset, in_query)
