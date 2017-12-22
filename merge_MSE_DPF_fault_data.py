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

header_file_name = 'MSE_header.csv'
#nrows_test 注意数量是不是批量的数据

#nrows_test = 5000000
nrows_test = None
source_file_name1 = 'MSE_new_feature_9month_feature_Bad.csv'  # 
source_file_name2 = 'MSE_new_feature_10month_feature_Bad.csv' #
source_file_name3 = 'MSE_new_feature_11month_feature_Bad.csv' #
source_file_name4 = 'DPF_new_feature_9month_feature_Bad.csv'  # 
source_file_name5 = 'DPF_new_feature_10month_feature_Bad.csv' #
source_file_name6 = 'DPF_new_feature_11month_feature_Bad.csv' #

experiment_type = 'concate_MSE_DPF_bad_data'
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
      header = None
      logging.warning('reading source file...')
      logging.warning('reading source file...' + str(source_file_name1))
      mse1= pd.read_csv(source_file_name1, names=header, nrows = nrows_test)
      logging.warning('reading source file...' + str(source_file_name2))
      mse2 = pd.read_csv(source_file_name2, names=header, nrows = nrows_test)
      logging.warning('reading source file...' + str(source_file_name3))
      mse3 = pd.read_csv(source_file_name3, names=header, nrows = nrows_test )
  
      logging.warning('reading source file...' + str(source_file_name4))
      dpf1 = pd.read_csv(source_file_name4, names=header, nrows = nrows_test)
      logging.warning('reading source file...' + str(source_file_name5))
      dpf2 = pd.read_csv(source_file_name5, names=header, nrows = nrows_test)
      logging.warning('reading source file...' + str(source_file_name6))
      dpf3 = pd.read_csv(source_file_name6, names=header, nrows = nrows_test )
     
      logging.warning('test 11')
#      good9['PreciseTimeStamp'] = pd.to_datetime(good9['PreciseTimeStamp'])
#      good10['PreciseTimeStamp'] = pd.to_datetime(bad10['PreciseTimeStamp'])
#      bad11['PreciseTimeStamp'] = pd.to_datetime(bad11['PreciseTimeStamp'])

      
#      good9_mode_dict = dict()
#      good10_mode_dict = dict()
#      good11_mode_dict = dict()
#      bad9_mode_dict = dict()
#      bad10_mode_dict = dict()      
#      bad11_mode_dict = dict()
#      logging.warning('getting mode of every columns in training data...')

##replace nan with mode      
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
      
#      good_disks8 = good9['SerialNumber'].unique()
#      good_disks9 = good10['SerialNumber'].unique()
#      good_disks10 = good11['SerialNumber'].unique()  
#      all_good = set(good_disks8)|set(good_disks9)|set(good_disks10)
#      
#      logging.warning('good disks 8:{} .'.format(len(good_disks8)))
#      logging.warning('good disks 9:{} .'.format(len(good_disks9)))
#      logging.warning('good disks 10:{} .'.format(len(good_disks10)))
#      logging.warning('all good disks :{} .'.format(len(all_good)))     
      
      mse_disks1 = mse1['SerialNumber'].unique()
      mse_disks2 = mse2['SerialNumber'].unique()
      mse_disks3 = mse3['SerialNumber'].unique()  
      all_mse= set(mse_disks1)|set(mse_disks2)|set(mse_disks3)
      logging.warning('mse disks 9:{} .'.format(len(mse_disks1)))
      logging.warning('mse disks 10:{} .'.format(len(mse_disks2)))
      logging.warning('mse disks 11:{} .'.format(len(mse_disks3)))
      logging.warning('all mse disks :{} .'.format(len(all_mse)))    
      
      dpf_disks1 = dpf1['SerialNumber'].unique()
      dpf_disks2 = dpf2['SerialNumber'].unique()
      dpf_disks3 = dpf3['SerialNumber'].unique()  
      all_dpf = set(dpf_disks1)|set(dpf_disks2)|set(dpf_disks3)     
      logging.warning('dpf disks 9:{} .'.format(len(dpf_disks1)))
      logging.warning('dpf disks 10:{} .'.format(len(dpf_disks2)))
      logging.warning('dpf disks 11:{} .'.format(len(dpf_disks3)))
      logging.warning('all dpf disks :{} .'.format(len(all_dpf)))         
      
      logging.warning('filter dpf disks.')      
      all_mse = set(all_mse)
      dpf_disks_filter1 = []
      dpf_disks_filter2 = []
      dpf_disks_filter3 = []
      for disk in dpf_disks1:
        if disk in all_mse:
          continue
        else :
          dpf_disks_filter1.append(disk)
      for disk in dpf_disks2:
        if disk in all_mse:
          continue
        else :
          dpf_disks_filter2.append(disk)    
      for disk in dpf_disks3:
        if disk in all_mse:
          continue
        else :
          dpf_disks_filter3.append(disk) 
      all_dpf_filter = set(dpf_disks_filter1)|set(dpf_disks_filter2)|set(dpf_disks_filter3)     
          
      dpf1 = dpf1[dpf1['SerialNumber'].isin(dpf_disks_filter1)]
      dpf2 = dpf2[dpf2['SerialNumber'].isin(dpf_disks_filter2)]
      dpf3 = dpf3[dpf3['SerialNumber'].isin(dpf_disks_filter3)]
 
      logging.warning('dpf disks_filter 9:{} .'.format(len(dpf_disks_filter1)))
      logging.warning('dpf disks_filter 10:{} .'.format(len(dpf_disks_filter2)))
      logging.warning('dpf disks_filter 11:{} .'.format(len(dpf_disks_filter3)))
      logging.warning('all dpf disks_filter :{} .'.format(len(all_dpf_filter)))    
      
      data1 = pd.concat([mse1, dpf1])
      data2 = pd.concat([mse2, dpf2])
      data3 = pd.concat([mse3, dpf3])

      all_disks1 = data1['SerialNumber'].unique()
      all_disks2 = data2['SerialNumber'].unique()
      all_disks3 = data3['SerialNumber'].unique()  
      all_disks_all= set(all_disks1)|set(all_disks2)|set(all_disks3)
      logging.warning('all_bad disks 9:{} .'.format(len(all_disks1)))
      logging.warning('all_bad disks 10:{} .'.format(len(all_disks2)))
      logging.warning('all_bad disks 11:{} .'.format(len(all_disks3)))
      logging.warning('all all_bad disks :{} .'.format(len(all_disks_all)))        

      data1.to_csv('all_bad9_features.csv', index=None)
      logging.warning('save all_bad9_features.')
      data2.to_csv('all_bad10_features.csv', index=None)
      logging.warning('save all_bad10_features.')
      data3.to_csv('all_bad11_features.csv', index=None)
      logging.warning('save all_bad11_features.')    
      
#      good9.to_csv('good9_somecolumn.csv', index=None)
#      logging.warning('save good9_somecolumn.')
#      good10.to_csv('good10_somecolumn.csv', index=None)
#      logging.warning('save good10_somecolumn.')
#      good11.to_csv('good11_somecolumn.csv', index=None)
#      logging.warning('save good11_somecolumn.')        
      
##good sample 
#      sample_disk8 = set()
#      sample_disk9 = set()
#      sample_disk10 = set()
#      
#      for single in good_disks8:
#          value =  random.randint(1, 100)
#          if value % 10 == 5:
#             sample_disk8.add(single)
#      for single in good_disks9:
#          value =  random.randint(1, 100)
#          if value % 10 == 5:
#             sample_disk9.add(single)
#      for single in good_disks10:
#          value =  random.randint(1, 100)
#          if value % 10 == 5:
#             sample_disk10.add(single)
#      
#      logging.warning('sample_disk8:{} .'.format(len(sample_disk8)))
#      logging.warning('sample_disk9:{} .'.format(len(sample_disk9)))
#      logging.warning('sample_disk10:{} .'.format(len(sample_disk10)))
#      logging.warning('all sample disks :{} .'.format(len(sample_disk8|sample_disk9|sample_disk10)))             
#
#      import csv
#      cw = csv.writer(open("sample_disk8_SerialNumber.csv",'w'))
#      cw.writerow(list(sample_disk8))
#      cw = csv.writer(open("sample_disk9_SerialNumber.csv",'w'))
#      cw.writerow(list(sample_disk9))
#      cw = csv.writer(open("sample_disk10_SerialNumber.csv",'w'))
#      cw.writerow(list(sample_disk10))      
#        
#      good_sample8 = good9[good9['SerialNumber'].isin(sample_disk8)]
#      good_sample9 = good10[good10['SerialNumber'].isin(sample_disk9)]
#      good_sample10 = good11[good11['SerialNumber'].isin(sample_disk10)]      
#      
#      good_sample8.to_csv('good_sample8.csv', index=None)
#      logging.warning('save good_sample8.')
#      good_sample9.to_csv('good_sample9.csv', index=None)
#      logging.warning('save good_sample9.')
#      good_sample10.to_csv('good_sample10.csv', index=None)
#      logging.warning('save good_sample10.')      
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