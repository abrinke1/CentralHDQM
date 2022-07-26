
VERBOSE = True

import os, sys
import timeit
sys.path.insert(1, os.path.realpath(os.path.pardir))
import db_access
from cern_sso import get_cookies
from collections import defaultdict
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if VERBOSE: print('\nInside test_local_app.py, finished imports')

### -------------------------------------------------------------------------------------------------- ###

def data(subsystem, pd, processing_string, from_run, to_run, runs, latest, series, series_id, in_query):
  if VERBOSE: print('\nInside data(%s, %s, %s, %d, %d, %s, %s, %s, %s, %s)' % \
                    (subsystem, pd, processing_string, from_run, to_run, \
                     runs, latest, series, series_id, in_query))

  if series_id == None:
    if subsystem == None:
      print('\nPlease provide a subsystem parameter.'); sys.exit()

    if pd == None:
      print('\nPlease provide a pd parameter.'); sys.exit()

    if processing_string == None:
      print('\nPlease provide a processing_string parameter.'); sys.exit()

  modes = 0
  if from_run != None and to_run != None: modes += 1
  if latest != None: modes += 1
  if runs != None: modes += 1

  if modes > 1:
    print('\nThe combination of parameters you provided is invalid.'); sys.exit()

  if runs != None:
    try:
      runs = runs.split(',')
      runs = [int(x) for x in runs]
    except:
      print('\nruns parameter is not valid. It has to be a comma separated list of integers.'); sys.exit()

  if series and series_id:
    print('\nseries and series_id can not be defined at the same time.'); sys.exit()

  db_access.setup_db()
  session = db_access.get_session()

  # Get series data by series_id
  if series_id:
    sql = '''
    SELECT selection_params.subsystem, selection_params.pd, selection_params.processing_string, last_calculated_configs.name
    FROM selection_params
    JOIN last_calculated_configs ON config_id = last_calculated_configs.id
    WHERE selection_params.id = :id
    ;
    '''

    rows = execute_with_retry(session, sql, { 'id': series_id })
    rows = list(rows)

    subsystem = rows[0]['subsystem']
    pd = rows[0]['pd']
    processing_string = rows[0]['processing_string']
    series = rows[0]['name']
  
  if runs == None:
    # Will need filtering
    runs_filter = ''
    latest_filter = ''
    if from_run != None and to_run != None:
      runs_filter = 'AND run >= %s AND run <= %s' % (from_run, to_run)
    else:
      if latest == None:
        latest = 50
      latest_filter = 'LIMIT %s' % latest

    run_class_like = '%%collision%%'
    if 'cosmic' in pd.lower():
      run_class_like = '%%cosmic%%'

    # UL2018 processing produced two processing strings: 12Nov2019_UL2018 and 12Nov2019_UL2018_rsb.
    # _rsb version is a resubmition because some runs were not processed (crashed?) in the initial processing.
    # Some runs might exist under both processing strings, some under just one of them!
    # A union of both of these processing strings contains all runs of UL2018.
    # So in HDQM, when 12Nov2019_UL2018 is requested, we need include 12Nov2019_UL2018_rsb as well!!!
    # This is a special case and is not used in any other occasion.

    processing_string_sql = 'AND processing_string=:processing_string'
    if processing_string == '12Nov2019_UL2018':
      processing_string_sql = 'AND (processing_string=:processing_string OR processing_string=:processing_string_rsb)'

    sql = '''
    SELECT DISTINCT run FROM oms_data_cache
    WHERE run IN 
    (
      SELECT run FROM historic_data_points
      WHERE subsystem=:subsystem
      AND pd=:pd
      %s
    )
    AND oms_data_cache.run_class %s :run_class
    AND oms_data_cache.significant=%s
    AND oms_data_cache.is_dcs=%s
    %s
    ORDER BY run DESC
    %s
    ;
    ''' % (processing_string_sql, db_access.ilike_crossdb(), db_access.true_crossdb(), db_access.true_crossdb(), runs_filter, latest_filter)

    print('Getting the list of runs...')
    start = timeit.default_timer() 

    rows = execute_with_retry(session, sql, { 
      'subsystem': subsystem, 
      'pd': pd, 
      'processing_string': processing_string, 
      'processing_string_rsb': processing_string + '_rsb', 
      'run_class': run_class_like 
    })
    rows = list(rows)

    stop = timeit.default_timer()
    print('Runs retrieved in: ', stop - start)

    runs = [x[0] for x in rows]

  # Construct SQL query
  query_params  = { 'subsystem': subsystem, 'pd': pd, 'processing_string': processing_string, 'processing_string_rsb': processing_string + '_rsb' }

  run_selection_sql = 'AND historic_data_points.run BETWEEN :from_run AND :to_run'
  if runs != None and len(runs) != 0:
    run_selection_sql = 'AND historic_data_points.run IN (%s)' % ', '.join(str(x) for x in runs)
    query_params['runs'] = runs
  else:
    query_params['from_run'] = from_run
    query_params['to_run'] = to_run

  series_filter_sql = ''
  if series != None:
    series_filter_sql = 'AND historic_data_points.name IN ('
    series = series.split(',')
    for i in range(len(series)):
      key = 'series_%i' % i
      series_filter_sql += ':%s,' % key
      query_params[key] = series[i]
    series_filter_sql = series_filter_sql.rstrip(',') + ')'

  processing_string_sql = 'AND historic_data_points.processing_string=:processing_string'
  if processing_string == '12Nov2019_UL2018':
    processing_string_sql = 'AND (historic_data_points.processing_string=:processing_string OR historic_data_points.processing_string=:processing_string_rsb)'

  sql = '''
  SELECT 
  historic_data_points.id,
	historic_data_points.run, 
	historic_data_points.value, 
	historic_data_points.error,
	historic_data_points.name, 
	
	historic_data_points.plot_title, 
	historic_data_points.y_title
  FROM historic_data_points

  WHERE historic_data_points.subsystem=:subsystem
  AND historic_data_points.pd=:pd
  %s

  %s
  %s

  ORDER BY historic_data_points.run ASC
  ;
  ''' % (processing_string_sql, run_selection_sql, series_filter_sql)

  print('Getting the data...')
  start = timeit.default_timer() 

  rows = execute_with_retry(session, sql, query_params)
  rows = list(rows)
  session.close()

  stop = timeit.default_timer()
  print('Data retrieved in: ', stop - start)

  result = {}
  for row in rows:
    # Names are unique within the subsystem
    key = '%s_%s' % (row['name'], subsystem)
    if key not in result:
      result[key] = {
        'metadata': { 
          'y_title': row['y_title'], 
          'plot_title': row['plot_title'], 
          'name': row['name'], 
          'subsystem': subsystem, 
          'pd': pd,
          'processing_string': processing_string,
        },
        'trends': []
      }

    # Because of the 12Nov2019_UL2018_rsb hack, a run might exist in both 
    # 12Nov2019_UL2018 and 12Nov2019_UL2018_rsb processing strings. This check
    # adds the run to the result only if it doesn't exist yet to avoid duplications.
    run_exists = next((x for x in result[key]['trends'] if x['run'] == row['run']), False)
    if not run_exists:
      result[key]['trends'].append({
        'run': row['run'],
        'value': row['value'],
        'error': row['error'],
        'id': row['id'],
        'oms_info': {},
      })

  # Transform result to array
  result = [result[key] for key in sorted(result.keys())]
  result = add_oms_info_to_result(result)

  # if VERBOSE: print(result)
  if VERBOSE: print('\nSaving trends to plots')
  for res in result:
    met = res['metadata']
    trd = res['trends']
    fig = plt.figure()
    x_val = np.array([int(tr['run']) for tr in trd])
    y_val = np.array([float(tr['value']) for tr in trd])
    y_err = np.array([float(tr['error']) for tr in trd])
    plt.scatter(x_val, y_val)
    plt.errorbar(x_val, y_val, yerr=y_err, fmt='o')
    plt.title(met['plot_title'])
    plt.xlabel('Run number')
    plt.ylabel(met['y_title'])
    fig.savefig('output/%s_%s_%s.pdf' % (met['subsystem'], met['pd'], met['name']))
    if VERBOSE: print('Saved output/%s_%s_%s.pdf' % (met['subsystem'], met['pd'], met['name']))

## End def data()


def add_oms_info_to_result(result):
  if VERBOSE: print('\nInside add_oms_info_to_result()')
  runs = set()
  for item in result:
    for trend in item['trends']:
      runs.add(trend['run'])
  runs = list(runs)

  db_access.dispose_engine()
  session = db_access.get_session()
    
  query = session.query(db_access.OMSDataCache)\
    .filter(db_access.OMSDataCache.run.in_(tuple(runs)))\
    .all()
  db_oms_data = list(query)
  session.close()

  oms_data_dict = defaultdict(list)
  for row in db_oms_data:
    oms_data_dict[row.run] = {
      'start_time': row.start_time,
      'end_time': row.end_time,
      'b_field': row.b_field,
      'energy': row.energy,
      'delivered_lumi': row.delivered_lumi,
      'end_lumi': row.end_lumi,
      'recorded_lumi': row.recorded_lumi,
      'l1_key': row.l1_key,
      'l1_rate': row.l1_rate,
      'hlt_key': row.hlt_key,
      'hlt_physics_rate': row.hlt_physics_rate,
      'duration': row.duration,
      'fill_number': row.fill_number,
      'injection_scheme': row.injection_scheme,
      'era': row.era,
    }

  # Add oms_info to the respose
  for item in result:
    for trend in item['trends']:
      trend['oms_info'] = oms_data_dict[trend['run']]

  return result


def execute_with_retry(session, sql, params=None):
  if VERBOSE: print('\nInside execute_with_retry()')
  try:
    result = session.execute(sql, params)
  except:
    print('\nInside execute_with_retry, retrying.')
    session = db_access.get_session()
    result = session.execute(sql)
  return result


### ------------------------------------------ ###

if __name__ == '__main__':
  if VERBOSE: print('\nInside test_local_app.py')

  parser = argparse.ArgumentParser(description='Local test of HDQM app.')
  parser.add_argument('--sys', dest='subsystem', type=str, default='L1T', help='Subsystem (list found in backend/extractor/cfg/)')
  parser.add_argument('--pd', dest='pd', type=str, default='SingleMuon', help='Primary dataset (ls /eos/cms/store/group/comm_dqm/DQMGUI_data/Run2022)')
  parser.add_argument('--proc', dest='processing_string', type=str, default='PromptReco', help='Processing (e.g. PromptReco, Express, etc.)')
  parser.add_argument('--from_run', dest='from_run', type=int, default=None, help='First run to process')
  parser.add_argument('--to_run', dest='to_run', type=int, default=None, help='Last run to process')
  parser.add_argument('--runs', dest='runs', type=str, default=None, help='List of runs to process, separated by commas, no spaces')
  parser.add_argument('--latest', dest='latest', type=str, default=None, help='Run over the latest N runs')
  parser.add_argument('--series', dest='series', type=str, default=None, help='Not sure what this is')
  parser.add_argument('--series_id', dest='series_id', type=str, default=None, help='Not sure what this is either')
  parser.add_argument('--query', dest='in_query', type=str, default=None, help='Trend metric query')

  args = parser.parse_args()

  if args.from_run or args.runs:
    from_run = args.from_run
    to_run = args.to_run
    runs = args.runs
  else:
    from_run = 0
    to_run = 999999
    runs = None

  if VERBOSE: print('\nAbout to run data()')
  data(args.subsystem, args.pd, args.processing_string, from_run, to_run, runs, \
       args.latest, args.series, args.series_id, args.in_query)
  if VERBOSE: print('\nRan data() - finished!')


