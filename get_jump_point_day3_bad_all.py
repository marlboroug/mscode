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

source_file_name1 = '10_10_HWWEFault_OLD_dropLastDay_Z1X_3month.csv'  #8 
source_file_name2 = '10_10_HWWEFault_OLD_dropLastDay_Z1X_2month.csv' #9
source_file_name3 = '10_10_HWWEFault_OLD_dropLastDay_Z1X_1month.csv' #10

source_file_name4 = '10_10_HWWEGood_OLD_dropLastDay_Z1X_3month.csv'  #8 
source_file_name5 = '10_10_HWWEGood_OLD_dropLastDay_Z1X_2month.csv' #9
source_file_name6 = '10_10_HWWEGood_OLD_dropLastDay_Z1X_1month.csv'   #10

experiment_type = 'get_jump_point_bad_compare_good'

year = 2016

def _get_experiment_base(fault_train_month, fault_test_month, good_month_start, good_month_stop, good_training_ratio,
  fault_disk_take_top, good_disk_take_top):
  return 'FTRM_{}-FTEM_{}-GM_{}_{}-GTR_{}-F_top_{}-G_top_{}'.format(
  fault_train_month, fault_test_month, good_month_start, good_month_stop, round(good_training_ratio, 2),
  fault_disk_take_top, good_disk_take_top
  )


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

#max min mean(两头除去0.5%) nums
def get_nums(data):
      #filter 1% data min or max because of noise
      allnum = len(data)
      if allnum < 1:
          return int(-1),int(-1),int(-1),int(-1)
          
      data_max =max(data)
      data_min = min(data)
    
      num_drop = 5
      if int(allnum / 200) > num_drop:
         num_drop = int(allnum / 200)
      
      data = sorted(data)
      data = data[num_drop:(allnum - num_drop)]

      
      if len(data) < 1:
          return int(-1),int(-1),int(-1),int(-1)
      
      mean = sum(data) / len(data)
      return data_max,data_min,mean,allnum

add_num = 'Top'
def preprocessingGroup2(group):
    # logging.warning("id "+str(gid))
    indexs = group.index.values
    num = len(indexs)
    for i in range(0, len(indexs), 1):
        group.set_value(indexs[i], add_num, num - i)
    return group

def multiProcPreprocess2(dfgb, keys):
    multiprocessing.freeze_support()
    #cpus = multiprocessing.cpu_count()
    cpus = 32
    results = []
    with multiprocessing.Pool(processes=cpus) as pool:
        for x in keys:
            result = pool.apply_async(preprocessingGroup2, args=(dfgb.get_group(x),))
            results.append(result)
        data = pd.concat([result.get() for result in results])
    logging.warning('multiprocess2 end one cycle')
#    results = []
#    #to test serial version
#    for x in keys:
#        result = preprocessingGroup2(dfgb.get_group(x))
#        results.append(result)
#    data = pd.concat([result for result in results])
#    logging.warning('multiprocess2 end one cycle')       
    return data

def draw_graph(good_data1,good_data2,bad_data,data4,feature_name):
      plt.cla()
      plt.close('all')
      x1=range(1,len(good_data1) + 1)
      x2=range(1,len(good_data2) + 1)
      x3=range(1,len(bad_data) + 1)
      x4=range(1,len(data4) + 1)
      plt.plot(x1,list(good_data1),label='data1',marker='o',color='r')
      plt.plot(x2,list(good_data2),label='data2',marker='o',color='g',)
      plt.plot(x3,list(bad_data),label='data3',marker='o',color='blue',)
      plt.plot(x4,list(data4),label='data4',marker='o',color='black',)     
      plt.xlabel(feature_name)
      plt.ylabel('mean value')
      plt.title('Compare Graph\n' + feature_name + "\n")
      plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # plt.show()
      folder = "picture_good_bad_29"
      if not os.path.exists(folder):
          logging.warning('create result folder {}.'.format(folder))
          os.makedirs(folder)
      plt.savefig(folder +  "/" + feature_name + '.jpg')
      plt.cla()
      plt.close('all')
      logging.warning('save feature picture:{} '.format(str(feature_name)))

def analyse_result(good_data1,good_data2,bad_data,data4,feature):
      good1 = []
      good2 = []
      bad = []
      result4 = []
      logging.warning('analyse feature picture:{} '.format(str(feature)))
      for i in range (29,0,-1):
    
          data_max,data_min,data_mean,data_nums = get_nums(good_data1[good_data1[add_num] == i][feature].tolist())
          logging.warning('data1 day:{}, mean:{}, nums:{}, max:{}, min:{}'.format(i, data_mean, data_nums, data_max,data_min))
          good1.append(data_mean)
            
          data_max,data_min,data_mean,data_nums = get_nums(good_data2[good_data2[add_num] == i][feature].tolist())
          logging.warning('data2 day:{}, mean:{}, nums:{}, max:{}, min:{}'.format(i, data_mean, data_nums, data_max,data_min))
          good2.append(data_mean)
    
          data_max,data_min,data_mean,data_nums = get_nums(bad_data[bad_data[add_num] == i][feature].tolist())
          logging.warning('data3 day:{}, mean:{}, nums:{}, max:{}, min:{}'.format(i, data_mean,data_nums,data_max,data_min))
          bad.append(data_mean)

          data_max,data_min,data_mean,data_nums = get_nums(data4[data4[add_num] == i][feature].tolist())
          logging.warning('data4 day:{}, mean:{}, nums:{}, max:{}, min:{}'.format(i, data_mean,data_nums,data_max,data_min))
          result4.append(data_mean)          
          
      draw_graph(good1,good2,bad,result4,feature)


def generate_data(fault_train_month, fault_test_month, good_month_start, good_month_stop, good_training_ratio,fault_disk_take_top, good_disk_take_top, source_file_name):
    #取出两个月的内容
      with open(header_file_name) as f:
          header = f.readline().strip('\n').split(',')
      
      logging.warning('reading source file...')
      logging.warning('reading source file...' + str(source_file_name1))
      bad8 = pd.read_csv(source_file_name1, names=header, nrows = nrows_test)
      logging.warning('reading source file...' + str(source_file_name2))
      bad9 = pd.read_csv(source_file_name2, names=header, nrows = nrows_test)
      logging.warning('reading source file...' + str(source_file_name3))
      bad10 = pd.read_csv(source_file_name3, names=header, nrows = nrows_test )
    
      logging.warning('reading source file...' + str(source_file_name4))
      good8 = pd.read_csv(source_file_name4, names=header, nrows = nrows_test)
      logging.warning('reading source file...' + str(source_file_name5))
      good9 = pd.read_csv(source_file_name5, names=header, nrows = nrows_test)
      logging.warning('reading source file...' + str(source_file_name6))
      good10 = pd.read_csv(source_file_name6, names=header, nrows = nrows_test )
       
      logging.warning('test 11')
      
      bad8['PreciseTimeStamp'] = pd.to_datetime(bad8['PreciseTimeStamp'])
      bad9['PreciseTimeStamp'] = pd.to_datetime(bad9['PreciseTimeStamp'])
      bad10['PreciseTimeStamp'] = pd.to_datetime(bad10['PreciseTimeStamp'])

      logging.warning('test 12')     
#      good8['PreciseTimeStamp'] = pd.to_datetime(good8['PreciseTimeStamp'])
#      good9['PreciseTimeStamp'] = pd.to_datetime(good9['PreciseTimeStamp'])      
#      good10['PreciseTimeStamp'] = pd.to_datetime(good10['PreciseTimeStamp'])      
      
      fault_disks8 = bad8['SerialNumber'].unique()
      fault_disks9 = bad9['SerialNumber'].unique()
      fault_disks10 = bad10['SerialNumber'].unique()  
      all_fault = set(fault_disks8)|set(fault_disks9)|set(fault_disks10)

      good_disks8 = good8['SerialNumber'].unique()
      good_disks9 = good9['SerialNumber'].unique()
      good_disks10 = good10['SerialNumber'].unique()  
      all_good = set(good_disks8)|set(good_disks9)|set(good_disks10)      
      
      logging.warning('fault disks 8:{} .'.format(len(fault_disks8)))
      logging.warning('fault disks 9:{} .'.format(len(fault_disks9)))
      logging.warning('fault disks 10:{} .'.format(len(fault_disks10)))
      logging.warning('all fault disks :{} .'.format(len(all_fault)))      
      
      logging.warning('good disks 8:{} .'.format(len(good_disks8)))
      logging.warning('good disks 9:{} .'.format(len(good_disks9)))
      logging.warning('good disks 10:{} .'.format(len(good_disks10)))
      logging.warning('all good disks :{} .'.format(len(all_good)))     
      
      
      to_be_deleted = []
      for name in bad8.columns:
            if name in ['model', 'capacity_bytes','FaultTimeStamp','EventId','ParsingError','AAMFeature','APMFeature','ATASecurity','ATAVersion','Device','DeviceModel','FirmwareVersion','ModelFamily','RdLookAhead','RotationRate','SMARTSupport','SATAVersion','SectorSize','UserCapacity','WriteCache','WtCacheReorder']:
                to_be_deleted.append(name)
            elif name == "LabelId" or name == "SerialNumber" or name == 'PreciseTimeStamp' or  name == 'NodeId':
                continue
            elif  str(bad8[name].std()) == "nan": #bad8[name].mean == 0 or
                to_be_deleted.append(name)
    
      bad8 = bad8.drop(to_be_deleted, axis=1)
      bad9 = bad9.drop(to_be_deleted, axis=1)
      bad10 = bad10.drop(to_be_deleted, axis=1)
      
      good8 = good8.drop(to_be_deleted, axis=1)
      good9 = good9.drop(to_be_deleted, axis=1)
      good10 = good10.drop(to_be_deleted, axis=1)     
      
      
      logging.warning('deleted columns:{}'.format(to_be_deleted))
      logging.warning('keep columns:{}'.format(str(bad8.columns)))
      
      bad8_mean_dict = dict()
      bad9_mean_dict = dict()
      bad10_mean_dict = dict()
      
      good8_mean_dict = dict()
      good9_mean_dict = dict()
      good10_mean_dict = dict()      
      logging.warning('getting mean of every columns in training data...')
      for name in bad8.columns:
          if name in ['NodeId', 'SerialNumber', 'PreciseTimeStamp','LabelId']:
              continue
          bad8_mean_dict[name] = bad8[name].mean()
          bad9_mean_dict[name] = bad9[name].mean()
          bad10_mean_dict[name] = bad10[name].mean()

          good8_mean_dict[name] = good8[name].mean()
          good9_mean_dict[name] = good9[name].mean()
          good10_mean_dict[name] = good10[name].mean()
 
      for name in bad8.columns:
          if name in ['NodeId', 'SerialNumber', 'PreciseTimeStamp','LabelId']:
              continue
          bad8[name][np.isnan(bad8[name])] = bad8_mean_dict[name]
          bad9[name][np.isnan(bad9[name])] = bad9_mean_dict[name]
          bad10[name][np.isnan(bad10[name])] = bad10_mean_dict[name]
          
          good8[name][np.isnan(good8[name])] = good8_mean_dict[name]
          good9[name][np.isnan(good9[name])] = good9_mean_dict[name]
          good10[name][np.isnan(good10[name])] = good10_mean_dict[name]          
          
          logging.warning('done replace nan with mean in training and testing column {}'.format(name))

      
      bad8[add_num] = 0
      bad9[add_num] = 0
      bad10[add_num] = 0
      
      good8[add_num] = 0
      good9[add_num] = 0
      good10[add_num] = 0
      
      badall = pd.concat([bad8,bad9,bad10])
      
      logging.warning('test:{} '.format(1))
      
      badall = multiProcPreprocess2(badall.groupby('SerialNumber'),badall['SerialNumber'].unique())
      good8 = multiProcPreprocess2(good8.groupby('SerialNumber'),good8['SerialNumber'].unique())
      good9 = multiProcPreprocess2(good9.groupby('SerialNumber'),good9['SerialNumber'].unique())
      good10 = multiProcPreprocess2(good10.groupby('SerialNumber'),good10['SerialNumber'].unique())      
      
      badall.to_csv('badall.csv', index=None)
      logging.warning('save badall.')
      
      good8.to_csv('good8.csv', index=None)
      logging.warning('save good8.')
      good9.to_csv('good9.csv', index=None)
      logging.warning('save good9.')
      good10.to_csv('good10.csv', index=None)
      logging.warning('save good10.')      
      
    
      logging.warning('test:{} '.format(2))
    # PreciseTimeStamp,SerialNumber,LabelId
      features = set(bad8.columns)
      features.remove('PreciseTimeStamp')
      features.remove('SerialNumber')
      features.remove('LabelId')
      features.remove(add_num)
      features.remove('NodeId')
      logging.warning('features :{} '.format(str(features)))
      
      for feature in features:
          logging.warning('feature :{} '.format(str(feature)))
          analyse_result(good8,good9,good10,badall,feature)


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


#代码的想法;通过均值画图分析，找出跳变的异常点，用来做坏盘的窗口
#在数据上对每一个唯独都去掉了top 0.5% 和最小值的0.5%
#理想情况下可以找出一个相对合适天数跳变点

#need to change "year", "start" "stop" 注意数量是不是批量的数据