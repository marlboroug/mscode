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
import matplotlib.mlab as mlab

header_file_name = 'header.csv'
# 注意数量是不是批量的数据
source_file_name8_train = "test_get_cv_result_sampleall_month_"+ "8" +"_featureid_3_GTR_ratio_0.7_FTR_ratio_0.7_Top_3_train.csv"
source_file_name8_test = "test_get_cv_result_sampleall_month_"+ "8" + "_featureid_3_GTR_ratio_0.3_FTR_ratio_0.3_Top_3_test.csv"

source_file_name9_train = "test_get_cv_result_sampleall_month_"+ "9" +"_featureid_3_GTR_ratio_0.7_FTR_ratio_0.7_Top_3_train.csv"
source_file_name9_test = "test_get_cv_result_sampleall_month_"+ "9" + "_featureid_3_GTR_ratio_0.3_FTR_ratio_0.3_Top_3_test.csv"

source_file_name10_train = "test_get_cv_result_sampleall_month_"+ "10" +"_featureid_3_GTR_ratio_0.7_FTR_ratio_0.7_Top_3_train.csv"
source_file_name10_test = "test_get_cv_result_sampleall_month_"+ "10" + "_featureid_3_GTR_ratio_0.3_FTR_ratio_0.3_Top_3_test.csv"

experiment_type = 'get_data_disturbtion_draw_picture'

nrows_test = None
year = 2016

def _get_experiment_base(fault_train_month, fault_test_month, good_month_start, good_month_stop, good_training_ratio,
  fault_disk_take_top, good_disk_take_top):
  return 'get_data_disturbtion_draw_picture'.format(
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

#(两头除去10%) nums
def get_filter_data(data,x_name,month,good_or_bad):
    import heapq
    data = list(data)
    bar = 100
    if len(data) / 50 > bar:
        bar = len(data) / 50
    datamax = heapq.nlargest(int(bar), data)
    datamin = heapq.nsmallest(int(bar), data)
    index = int(bar) - 1
    logging.warning('{}_{}_{} data init len:{},delete bar nums:{} max value:{}, delete max bar value:{}, min value:{} , delete min bar value:{}'.format(
            x_name,month,good_or_bad,len(data), bar, datamax[0],datamax[int(bar) - 1],datamin[0],datamin[int(bar) - 1]))
    data = list(filter(lambda x: ((x >= datamin[index]) & (x <= datamax[index])) , data))
    return data

def draw_graph(lst,x_name,month,good_or_bad, feature_type):
    plt.cla()
    plt.close('all')
    mu = np.mean(lst)  # 样本均值
    sigma = np.std(lst)  # 样本标准差
    num_bins = 100  # 区间数量(实际样本值是从100到150, 所以分成50份)
    n, bins, patches = plt.hist(lst, num_bins, normed=1, facecolor='blue', alpha=0.5)
    # 添加一个理想的正态分布拟合曲线
    y = mlab.normpdf(bins, mu, sigma)
    plt.plot(bins, y, 'r--')
    plt.xlabel(x_name)
    plt.ylabel('Probability')
    plt.title(r'$\mu={}$, $\sigma={}$'.format(round(mu, 3), round(sigma, 3)))
    plt.subplots_adjust(left=0.15)
#    plt.show()  
    folder =  "picture_data_disturbtion_" + feature_type + "_" + str(month) + "_" + str(good_or_bad) 
    if not os.path.exists(folder):
      logging.warning('create result folder {}.'.format(folder))
      os.makedirs(folder)
    file_name = str(month) + "_" + good_or_bad + "_" + x_name  + '.jpg'
    plt.savefig(folder +  "/" + file_name)
    plt.cla()
    plt.close('all')
    logging.warning('save feature picture:{}-{}-{} '.format(str(month),good_or_bad,x_name))    

def generate_data(fault_train_month, fault_test_month, good_month_start, good_month_stop, good_training_ratio,fault_disk_take_top, good_disk_take_top, source_file_name):

      header = None
      logging.warning('reading source file...')
      logging.warning('reading source file...' + str(source_file_name8_train))
      train8 = pd.read_csv(source_file_name8_train, names= header, nrows = nrows_test)
      logging.warning('reading source file...' + str(source_file_name8_test))
      test8 = pd.read_csv(source_file_name8_test, names= header, nrows = nrows_test)
      
      logging.warning('reading source file...' + str(source_file_name9_train))
      train9 = pd.read_csv(source_file_name9_train, names= header, nrows = nrows_test)
      logging.warning('reading source file...' + str(source_file_name9_test))
      test9 = pd.read_csv(source_file_name9_test, names= header, nrows = nrows_test)     
      
      logging.warning('reading source file...' + str(source_file_name10_train))
      train10 = pd.read_csv(source_file_name10_train, names= header, nrows = nrows_test)
      logging.warning('reading source file...' + str(source_file_name10_test))
      test10 = pd.read_csv(source_file_name10_test, names= header, nrows = nrows_test)   
    
      data8 = pd.concat([train8,test8])  
      data9 = pd.concat([train9,test9])  
      data10 = pd.concat([train10,test10])  
      
      data8_good = data8[data8['LabelId'] == 0]
      data8_bad = data8[data8['LabelId'] == 1]
      
      data9_good = data9[data9['LabelId'] == 0]
      data9_bad = data9[data9['LabelId'] == 1]
      
      data10_good = data10[data10['LabelId'] == 0]
      data10_bad = data10[data10['LabelId'] == 1]   

      with open(header_file_name) as f:
          header = f.readline().strip('\n').split(',')
        
      logging.warning('test:{} '.format(2))
    # PreciseTimeStamp,SerialNumber,LabelId
      features = set(data8_bad.columns)
      features.remove('PreciseTimeStamp')
      features.remove('SerialNumber')
      features.remove('LabelId')
      features.remove('NodeId')
      logging.warning('features :{} '.format(str(features)))
     
      for feature in features:
          logging.warning('feature :{} '.format(str(feature)))
 
#      data8_good[list(features)] = data8_good[list(features)].fillna(0.0).astype(int)
#      data8_bad[list(features)] = data8_bad[list(features)].fillna(0.0).astype(int)
#      data9_good[list(features)] = data9_good[list(features)].fillna(0.0).astype(int)
#      data9_bad[list(features)] = data9_bad[list(features)].fillna(0.0).astype(int)
#      data10_good[list(features)] = data10_good[list(features)].fillna(0.0).astype(int)
#      data10_bad[list(features)] = data10_bad[list(features)].fillna(0.0).astype(int)
 
          data8_good[feature] = data8_good[feature].fillna(0.0).astype(int)
          data8_bad[feature] = data8_bad[feature].fillna(0.0).astype(int)
          data9_good[feature] = data9_good[feature].fillna(0.0).astype(int)
          data9_bad[feature] = data9_bad[feature].fillna(0.0).astype(int)
          data10_good[feature] = data10_good[feature].fillna(0.0).astype(int)
          data10_bad[feature] = data10_bad[feature].fillna(0.0).astype(int)

          data8_1 = list(data8_good[feature])
          data8_1 = get_filter_data(data8_1, feature, 8, "good")
          
          data8_2 = list(data8_bad[feature])
          data8_2 = get_filter_data(data8_2, feature, 8, "bad")
          
          data9_1 = list(data9_good[feature])
          data9_1 = get_filter_data(data9_1, feature, 9, "good")  
          
          data9_2 = list(data9_bad[feature])
          data9_2 = get_filter_data(data9_2, feature, 9, "bad")  
          
          data10_1 = list(data10_good[feature])
          data10_1 = get_filter_data(data10_1, feature, 10, "good")  
          
          data10_2 = list(data10_bad[feature])
          data10_2 = get_filter_data(data10_2, feature, 10, "bad")  
          
          if feature in header:
             draw_graph(data8_1,feature,8,"good","base")
             draw_graph(data8_2,feature,8,"bad","base")
             
             draw_graph(data9_1,feature,9,"good","base")
             draw_graph(data9_2,feature,9,"bad","base")
             
             draw_graph(data10_1,feature,10,"good","base")
             draw_graph(data10_2,feature,10,"bad","base")
          else :
             draw_graph(data8_1,feature,8,"good","up")
             draw_graph(data8_2,feature,8,"bad","up")
             
             draw_graph(data9_1,feature,9,"good","up")
             draw_graph(data9_2,feature,9,"bad","up")
             
             draw_graph(data10_1,feature,10,"good","up")
             draw_graph(data10_2,feature,10,"bad","up")              


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