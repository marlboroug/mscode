# encoding: utf-8
"""cross_validation_month_detail

Usage:
  cross_validation_month_fault_detail.py
  [--fault_disk_take_top=<fault_disk_take_top>] [--good_disk_take_top=<good_disk_take_top>] [--source_file_name=<source_file_name>] [--model_type=<model_type>]
  cross_validation_month_fault_detail.py (-h | --help)
  cross_validation_month_fault_detail.py --version

Options:
  --fault_disk_take_top=<fault_disk_take_top> epoch. [default: 10].
  --good_disk_take_top=<good_disk_take_top> epoch. [default: None].
  --source_file_name=<source_file_name> epoch. [default: backblaze_2016.csv].
  --model_type=<model_type> model. [default: RandomForest].
  -h --help Show this screen.
  --version Show version.
"""
# (--fault_train_month=<fault_train_month>) (--fault_test_month=<fault_test_month>) (--good_month_start=<good_month_start>) (--good_month_stop=<good_month_stop>) (--good_training_ratio=<good_training_ratio>)
from docopt import docopt
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from collections import Counter
import numpy as np
import logging
from calendar import monthrange
import datetime
import _pickle as cPickle
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import tree
import pydotplus
import random
import multiprocessing

header_file_name = 'header.csv'
#nrows_test 注意数量是不是批量的数据

nrows_test = None
#source_file_name1 = 'all_good9_somecolumn.csv'  #
#source_file_name2 = 'all_good10_somecolumn.csv' #
#source_file_name3 = 'all_good11_somecolumn.csv' #

source_file_name4 = '11_30_DPF_fault_9month.csv'  #
source_file_name5 = '11_30_DPF_fault_10month.csv' #
source_file_name6 = '11_30_DPF_fault_11month.csv' #

experiment_type = 'generate_sample_good_data_for_feature_engining'
year = 2016

def _get_experiment_base(fault_train_month, fault_test_month, good_month_start, good_month_stop, good_training_ratio,
  fault_disk_take_top, good_disk_take_top):
    return "file"
#  return 'FTRM_{}-FTEM_{}-GM_{}_{}-GTR_{}-F_top_{}-G_top_{}'.format(
#  fault_train_month, fault_test_month, good_month_start, good_month_stop, round(good_training_ratio, 2),
#  fault_disk_take_top, good_disk_take_top
#  )


def _get_train_prefix(fault_train_month, fault_test_month, good_month_start, good_month_stop, good_training_ratio,
  fault_disk_take_top, good_disk_take_top):
  return _get_experiment_base(fault_train_month, fault_test_month, good_month_start, good_month_stop,
  good_training_ratio,
  fault_disk_take_top, good_disk_take_top) + '_train'


def _get_test_prefix(fault_train_month, fault_test_month, good_month_start, good_month_stop, good_training_ratio,
  fault_disk_take_top, good_disk_take_top):
  return _get_experiment_base(fault_train_month, fault_test_month, good_month_start, good_month_stop,
  good_training_ratio,
  fault_disk_take_top, good_disk_take_top) + '_test'


def _get_train_file_name(fault_train_month, fault_test_month, good_month_start, good_month_stop, good_training_ratio,
  fault_disk_take_top, good_disk_take_top):
  return experiment_type + '_' + _get_train_prefix(fault_train_month, fault_test_month, good_month_start,
  good_month_stop, good_training_ratio,
  fault_disk_take_top, good_disk_take_top) + '.csv'


def _get_test_file_name(fault_train_month, fault_test_month, good_month_start, good_month_stop, good_training_ratio,
  fault_disk_take_top, good_disk_take_top):
  return experiment_type + '_' + _get_test_prefix(fault_train_month, fault_test_month, good_month_start,
  good_month_stop, good_training_ratio,
  fault_disk_take_top, good_disk_take_top) + '.csv'


def _get_result_folder(fault_train_month, fault_test_month, good_month_start, good_month_stop, good_training_ratio,
  fault_disk_take_top, good_disk_take_top):
  return experiment_type + '_' + _get_experiment_base(fault_train_month, fault_test_month, good_month_start,
  good_month_stop, good_training_ratio,
  fault_disk_take_top, good_disk_take_top)


def __get_first_day_in_month(month):
  return datetime.datetime.strptime(
  '{}-{}-01'.format(int(year),month), "%Y-%m-%d")


def __get_last_day_in_month(month):
  return datetime.datetime.strptime(
  '{}-{}-{}'.format(int(year),month, monthrange(int(year), month)[1]), "%Y-%m-%d")


def _do_statistics(df):
  y = df["LabelId"]
  fault_unique_disks = df[df['LabelId'] == 1]['SerialNumber'].unique()
  good_unique_disks = df[df['LabelId'] == 0]['SerialNumber'].unique()
  logging.warning(
  'good disks:{}, fault disks:{}, item statistic:{}'.format(
  len(good_unique_disks), len(fault_unique_disks),
  Counter(y)))


def generate_data(fault_train_month, fault_test_month, good_month_start, good_month_stop, good_training_ratio,fault_disk_take_top, good_disk_take_top, source_file_name):
    #取出两个月的内容
      with open(header_file_name) as f:
          header = f.readline().strip('\n').split(',')
      logging.warning('reading source file...')
      logging.warning('reading source file...' + str(source_file_name4))
      bad1 = pd.read_csv(source_file_name4, names=header, nrows = nrows_test)
      logging.warning('reading source file...' + str(source_file_name5))
      bad2 = pd.read_csv(source_file_name5, names=header, nrows = nrows_test)
      logging.warning('reading source file...' + str(source_file_name6))
      bad3 = pd.read_csv(source_file_name6, names=header, nrows = nrows_test )
      
      logging.warning('init data...')     
      logging.warning('file 1 : ')
      _do_statistics(bad1)
      logging.warning('file 2 : ')
      _do_statistics(bad2)
      logging.warning('file 3 : ')
      _do_statistics(bad3)
      
      
      logging.warning('test 11')
      bad1['PreciseTimeStamp'] = pd.to_datetime(bad1['PreciseTimeStamp'])
      bad2['PreciseTimeStamp'] = pd.to_datetime(bad2['PreciseTimeStamp'])
      bad3['PreciseTimeStamp'] = pd.to_datetime(bad3['PreciseTimeStamp'])
#Min_DiskPartFailedTime,
#Max_DiskPartFailedTime,
      bad1['Min_DiskPartFailedTime'] = pd.to_datetime(bad1['Min_DiskPartFailedTime'])
      bad2['Min_DiskPartFailedTime'] = pd.to_datetime(bad2['Min_DiskPartFailedTime'])
      bad3['Min_DiskPartFailedTime'] = pd.to_datetime(bad3['Min_DiskPartFailedTime']) 


      bad1['Max_DiskPartFailedTime'] = pd.to_datetime(bad1['Max_DiskPartFailedTime'])
      bad2['Max_DiskPartFailedTime'] = pd.to_datetime(bad2['Max_DiskPartFailedTime'])
      bad3['Max_DiskPartFailedTime'] = pd.to_datetime(bad3['Max_DiskPartFailedTime']) 
      

      bad1 = bad1[bad1["Max_DiskPartFailedTime"] > bad1['PreciseTimeStamp']]
      bad2 = bad2[bad2["Max_DiskPartFailedTime"] > bad2['PreciseTimeStamp']]
      bad3 = bad3[bad3["Max_DiskPartFailedTime"] > bad3['PreciseTimeStamp']]

      logging.warning('after filter data...')     
      logging.warning('file 1 : ')
      _do_statistics(bad1)
      logging.warning('file 2 : ')
      _do_statistics(bad2)
      logging.warning('file 3 : ')
      _do_statistics(bad3)    
      
      bad1.to_csv('filter_11_30_DPF_fault_9month.csv', index=None)
      logging.warning('save filter_11_30_DPF_fault_9month.')
      bad2.to_csv('filter_11_30_DPF_fault_10month.csv', index=None)
      logging.warning('save filter_11_30_DPF_fault_10month.')
      bad3.to_csv('filter_11_30_DPF_fault_11month.csv', index=None)
      logging.warning('save filter_11_30_DPF_fault_11month.')         
      
#drop some columns but not need     
      to_be_deleted = []
      for name in bad1.columns:
            if name in ['TakenOut','Min_DiskPartFailedTime','Max_DiskPartFailedTime','FailureCategory','model', 'capacity_bytes','FaultTimeStamp','EventId','ParsingError','AAMFeature','APMFeature','ATASecurity','ATAVersion','Device','DeviceModel','FirmwareVersion','ModelFamily','RdLookAhead','RotationRate','SMARTSupport','SATAVersion','SectorSize','UserCapacity','WriteCache','WtCacheReorder']:
                to_be_deleted.append(name)
            elif name  in ["LabelId","SerialNumber","PreciseTimeStamp","NodeId" ]: 
                continue
            elif  str(bad1[name].std()) == "nan": #bad9[name].mean == 0 or
                to_be_deleted.append(name)

      bad1 = bad1.drop(to_be_deleted, axis=1)
      bad2 = bad2.drop(to_be_deleted, axis=1)
      bad3 = bad3.drop(to_be_deleted, axis=1)     

      bad1.to_csv('filter_11_30_DPF_fault_9month_some_columns.csv', index=None)
      logging.warning('save filter_11_30_DPF_fault_9month_some_columns.')
      bad2.to_csv('filter_11_30_DPF_fault_10month_some_columns.csv', index=None)
      logging.warning('save filter_11_30_DPF_fault_10month_some_columns.')
      bad3.to_csv('filter_11_30_DPF_fault_11month_some_columns.csv', index=None)
      logging.warning('save filter_11_30_DPF_fault_11month_some_columns.')    
      
      logging.warning('deleted columns:{}'.format(to_be_deleted))
      logging.warning('keep columns:{}'.format(str(bad1.columns)))
      #get all data


##replace nan with mode    
            
#      good9_mode_dict = dict()
#      good10_mode_dict = dict()
#      good11_mode_dict = dict()
#      
#      bad9_mode_dict = dict()
#      bad10_mode_dict = dict()      
#      bad11_mode_dict = dict()
#
#      
#      logging.warning('getting mode of every columns in training data...')
#      for name in bad9.columns:
#          if name in ['Min_DiskPartFailedTime','Max_DiskPartFailedTime','NodeId', 'SerialNumber', 'PreciseTimeStamp','LabelId']:
#              continue
#          
#          good9_mode_dict[name] = good9[name].mode()
#          good10_mode_dict[name] = good10[name].mode()
#          good11_mode_dict[name] = good11[name].mode()
#          
#          bad10_mode_dict[name] = bad10[name].mode()
#          bad11_mode_dict[name] = bad11[name].mode()
#          bad11_mode_dict[name] = bad11[name].mode()          
# 
#      for name in bad9.columns:
#          if name in ['Min_DiskPartFailedTime','Max_DiskPartFailedTime','NodeId', 'SerialNumber', 'PreciseTimeStamp','LabelId']:
#              continue
#          
#          good9[name][np.isnan(good9[name])] = good9_mode_dict[name]
#          good10[name][np.isnan(good10[name])] = good10_mode_dict[name]
#          good11[name][np.isnan(good11[name])] = good11_mode_dict[name]          
#          
#          bad9[name][np.isnan(bad9[name])] = bad9_mode_dict[name]
#          bad10[name][np.isnan(bad10[name])] = bad10_mode_dict[name]
#          bad11[name][np.isnan(bad11[name])] = bad11_mode_dict[name]             
#          logging.warning('done replace nan with mode in training and testing column {}'.format(name))
##replace nan with mode
      
#      good_disks9 = good9['SerialNumber'].unique()
#      good_disks10 = good10['SerialNumber'].unique()
#      good_disks11 = good11['SerialNumber'].unique()  
#      all_good = set(good_disks9)|set(good_disks10)|set(good_disks11)
#      
#      logging.warning('good disks 8:{} .'.format(len(good_disks9)))
#      logging.warning('good disks 9:{} .'.format(len(good_disks10)))
#      logging.warning('good disks 10:{} .'.format(len(good_disks11)))
#      logging.warning('all good disks :{} .'.format(len(all_good)))     
      
#      bad_disks8 = bad9['SerialNumber'].unique()
#      bad_disks9 = bad10['SerialNumber'].unique()
#      bad_disks10 = bad11['SerialNumber'].unique()  
#      all_bad= set(bad_disks8)|set(bad_disks9)|set(bad_disks10)
      
#      logging.warning('bad disks 8:{} .'.format(len(bad_disks8)))
#      logging.warning('bad disks 9:{} .'.format(len(bad_disks9)))
#      logging.warning('bad disks 10:{} .'.format(len(bad_disks10)))
#      logging.warning('all bad disks :{} .'.format(len(all_bad)))         
#      
#      bad9.to_csv('bad9.csv', index=None)
#      logging.warning('save bad9.')
#      bad10.to_csv('bad10.csv', index=None)
#      logging.warning('save bad10.')
#      bad11.to_csv('bad11.csv', index=None)
#      logging.warning('save bad11.')    
      
#      good9.to_csv('good9.csv', index=None)
#      logging.warning('save good9.')
#      good10.to_csv('good10.csv', index=None)
#      logging.warning('save good10.')
#      good11.to_csv('good11.csv', index=None)
#      logging.warning('save good11.')        
      
##good sample 
#      sample_disk1 = set()
#      sample_disk2 = set()
#      sample_disk3 = set()
#      
#      for single in good_disks9:
#          value =  random.randint(1, 100)
#          if value % 10 == 5:
#             sample_disk1.add(single)
#      for single in good_disks10:
#          value =  random.randint(1, 100)
#          if value % 10 == 5:
#             sample_disk2.add(single)
#      for single in good_disks11:
#          value =  random.randint(1, 100)
#          if value % 10 == 5:
#             sample_disk3.add(single)
#      
#      logging.warning('sample_disk1:{} .'.format(len(sample_disk1)))
#      logging.warning('sample_disk2:{} .'.format(len(sample_disk2)))
#      logging.warning('sample_disk3:{} .'.format(len(sample_disk3)))
#      logging.warning('all sample disks :{} .'.format(len(sample_disk1|sample_disk2|sample_disk3)))             
#
#      import csv
#      cw = csv.writer(open("sample_good_disk9_SerialNumber.csv",'w'))
#      cw.writerow(list(sample_disk1))
#      cw = csv.writer(open("sample_good_disk10_SerialNumber.csv",'w'))
#      cw.writerow(list(sample_disk2))
#      cw = csv.writer(open("sample_good_disk11_SerialNumber.csv",'w'))
#      cw.writerow(list(sample_disk3))      
#        
#      good_sample1 = good9[good9['SerialNumber'].isin(sample_disk1)]
#      good_sample2 = good10[good10['SerialNumber'].isin(sample_disk2)]
#      good_sample3 = good11[good11['SerialNumber'].isin(sample_disk3)]      
#      
#      good_sample1.to_csv('good_sample9.csv', index=None)
#      logging.warning('save good_sample9.')
#      good_sample2.to_csv('good_sample10.csv', index=None)
#      logging.warning('save good_sample10.')
#      good_sample3.to_csv('good_sample11.csv', index=None)
#      logging.warning('save good_sample11.')      
#      logging.warning('test:{} '.format(2))

if __name__ == "__main__":
#  arguments = docopt(__doc__, version='cross_validation_month_detail 1.0')
    
      fault_train_month = [1,2,3]
      fault_test_month = [1,2,3]
      good_month_start = 1
      good_month_stop = 1
      good_training_ratio = float(0.5)
      fault_disk_take_top = 1
      good_disk_take_top = 1
      source_file_name = ''
      folder = _get_result_folder(fault_train_month, fault_test_month, good_month_start, good_month_stop,good_training_ratio,fault_disk_take_top, good_disk_take_top)
    
      if not os.path.exists(folder):
          logging.warning('create result folder {}.'.format(folder))
          os.makedirs(folder)
      else:
          logging.warning(
          'result folder {} exists! will overwrite files under it and appending logging file!'.format(folder))
          logging.basicConfig(level=logging.WARNING,format='%(asctime)-15s %(message)s')
      # set logging
      logging.warning('removing previous logging handler...')
      rootLogger = logging.getLogger()
      for h in rootLogger.handlers[1:]:
        rootLogger.removeHandler(h)
      logging.warning('adding new handler...')
      fileHandler = logging.FileHandler(filename=os.path.join(folder, 'logging.txt'))
      logFormatter = logging.Formatter('%(asctime)-15s %(message)s')
      fileHandler.setFormatter(logFormatter)
      rootLogger.addHandler(fileHandler)
    
      logging.warning('analyse data...')
      generate_data(fault_train_month, fault_test_month, good_month_start, good_month_stop, good_training_ratio,fault_disk_take_top, good_disk_take_top, source_file_name)


# used to sample good data and filter no use columns in all good and bad data