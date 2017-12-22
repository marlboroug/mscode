# encoding: utf-8
"""cross_validation_by_month
Usage:
    cross_validation_by_month.py   
               (--month=<month>) (--featureid=<featureid>) (--delete_feature_contain=<delete_feature_contain>) (--additional_feature=<additional_feature>)   
               [--fault_disk_take_top=<fault_disk_take_top>] [--fault_disk_drop_tail=<fault_disk_drop_tail>] [--source_file_name=<source_file_name>] [--model_type=<model_type>]
    cross_validation_by_month.py (-h | --help)
    cross_validation_by_month.py --version

Options:
  --fault_disk_take_top=<fault_disk_take_top> epoch.  [default: 1].
  --source_file_name=<source_file_name> epoch.  [default: backblaze_2015.csv].
  --model_type=<model_type> model.  [default: RandomForest].
  --fault_disk_drop_tail=<fault_disk_drop_tail> whether to drop tail fault data.  [default: False].
  -h --help     Show this screen.
  --version     Show version.
"""
#    (--good_training_ratio=<good_training_ratio>) (--fault_training_ratio=<fault_training_ratio>)       
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

# init need to split data 0.7 and 0.3, then don't need to do it 
month = 10
featureid = 2

source_file_name_train = "test_get_cv_result_sampleall_month_"+ str(month) +"_featureid_3_GTR_ratio_0.7_FTR_ratio_0.7_Top_3_train.csv"
source_file_name_test = "test_get_cv_result_sampleall_month_"+str(month) + "_featureid_3_GTR_ratio_0.3_FTR_ratio_0.3_Top_3_test.csv"
experiment_type = 'MSE_PDF_get_cv_result_small_month_' + str(month) + "_id_" + str(featureid)
delete_feature_contain =[
    "Event",
    "diff",
    "sigma"
]
additional_feature = [
#    "Event_15_sigma3",
#    "Event_129_sigma3",
#    "Event_129_diff4",
#    "Event_153_sigma3",
#    "Event_153_diff4",
"SmartAttribute_Reallocated_Event_Count_RAW_VALUE",
"SmartAttribute_Reallocated_Event_Count_VALUE",
#"Event_153",
        ]#

topnum = 3
nrows_test = None

#feature 
header_file_name = 'header_num.csv'
base_feature =[]
add_feature =[] #add feature

def _get_experiment_base(fault_month_start, fault_month_stop, good_month_start, good_month_stop, good_training_ratio, fault_training_ratio, fault_disk_take_top, fault_disk_drop_tail):
    return 'GTR_ratio_{}_FTR_ratio_{}_Top_{}'.format(round(good_training_ratio, 2),round(fault_training_ratio, 2), topnum)

def _get_train_prefix(fault_month_start, fault_month_stop, good_month_start, good_month_stop, good_training_ratio,  fault_training_ratio, fault_disk_take_top,fault_disk_drop_tail):
    return _get_experiment_base(fault_month_start, fault_month_stop, good_month_start, good_month_stop, good_training_ratio, fault_training_ratio, fault_disk_take_top, fault_disk_drop_tail) + '_train'

def _get_test_prefix(fault_month_start, fault_month_stop, good_month_start, good_month_stop, good_training_ratio, fault_training_ratio, fault_disk_take_top,fault_disk_drop_tail):
    return _get_experiment_base(fault_month_start, fault_month_stop, good_month_start, good_month_stop,1 - good_training_ratio, 1 - fault_training_ratio,fault_disk_take_top, fault_disk_drop_tail) + '_test'

def _get_train_file_name(fault_month_start, fault_month_stop, good_month_start, good_month_stop, good_training_ratio,fault_training_ratio, fault_disk_take_top,fault_disk_drop_tail):
    return experiment_type + '_' + _get_train_prefix(fault_month_start, fault_month_stop, good_month_start,good_month_stop, good_training_ratio, fault_training_ratio,fault_disk_take_top, fault_disk_drop_tail) + '.csv'

def _get_test_file_name(fault_month_start, fault_month_stop, good_month_start, good_month_stop, good_training_ratio,fault_training_ratio, fault_disk_take_top,fault_disk_drop_tail):
    return experiment_type + '_' + _get_test_prefix(fault_month_start, fault_month_stop, good_month_start,good_month_stop, good_training_ratio, fault_training_ratio,fault_disk_take_top, fault_disk_drop_tail) + '.csv'

def _get_result_folder(fault_month_start, fault_month_stop, good_month_start, good_month_stop, good_training_ratio,fault_training_ratio, fault_disk_take_top,fault_disk_drop_tail):
    return experiment_type + '_' + _get_experiment_base(fault_month_start, fault_month_stop, good_month_start,good_month_stop, good_training_ratio,fault_training_ratio, fault_disk_take_top, fault_disk_drop_tail)


def __get_first_day_in_month(month):
    return datetime.datetime.strptime(
        '2016-{}-01'.format(month), "%Y-%m-%d")


def __get_last_day_in_month(month):
    return datetime.datetime.strptime(
        '2016-{}-{}'.format(month, monthrange(2016, month)[1]), "%Y-%m-%d")

def draw_curve(pred, y, file_name_prefix):
    precision1, recall1, _ = metrics.precision_recall_curve(y, pred)
    f1_score = metrics.f1_score(y, pred.round())
    plt.clf()
    plt.plot(recall1, precision1, color='gold', label='class1-PR(F1-score {})'.format(f1_score))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.title('PR')
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(file_name_prefix + '_pr.png')
    logging.warning('saved {}_pr.png'.format(file_name_prefix))

    fpr, tpr, _ = metrics.roc_curve(y, pred)
    index_001 = None
    for i, (_fpr, _tpr) in enumerate(zip(fpr, tpr)):
        if _fpr >= 0.001:
            logging.warning('fpr:{}, tpr:{}.'.format(_fpr, _tpr))
            index_001 = i
            break
    auc = metrics.auc(fpr, tpr)
    plt.clf()
    plt.plot(fpr, tpr, color='gold', label='class1-ROC(AUC {})'.format(auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(file_name_prefix + '_roc.png')
    logging.warning('saved {}_roc.png'.format(file_name_prefix))

    fpr = fpr[:index_001]
    tpr = tpr[:index_001]
    plt.clf()
    plt.plot(fpr, tpr, color='gold', label='class1-ROC(AUC {})'.format(auc))
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.ylim([0.0, tpr[-1]])
    plt.xlim([0.0, fpr[-1]])
    plt.title('ROC-0.001')
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(file_name_prefix + '_roc_0.001.png')
    logging.warning('saved {}_roc_0.001.png'.format(file_name_prefix))


def _do_statistics(df):
    y = df["LabelId"]
    fault_unique_disks = df[df['LabelId'] == 1]['SerialNumber'].unique()
    good_unique_disks = df[df['LabelId'] == 0]['SerialNumber'].unique()
    logging.warning(
        'good disks:{}, fault disks:{}, item statistic:{}'.format(
            len(good_unique_disks), len(fault_unique_disks),
            Counter(y)))
    
def save_model_to_file(model, model_file_prefix, columns):
    with open(model_file_prefix + '.model', 'wb') as f:
        cPickle.dump(model, f)
        importance = model.feature_importances_
        importance = pd.DataFrame(importance, index=columns,
                                  columns=["Importance"])

        importance["Std"] = np.std([tree.feature_importances_
                                    for tree in model.estimators_], axis=0)
        importance.to_csv('{}_feature_importance.csv'.format(model_file_prefix))


def load_model_from_file_if_exists(model_file_prefix):
    if os.path.exists(model_file_prefix + '.model'):
        with open(model_file_prefix + '.model', 'rb') as f:
            model = cPickle.load(f)
            return model
    else:
        return None


class Model:
    def __init__(self):
        self.name = ''

    def get_new(self):
        pass

    def do_after_train(self, model, *args, **kwargs):
        pass


class RFModel:
    def __init__(self):
        self.name = 'RandomForest'

    def get_new(self):
        return RandomForestClassifier(n_estimators=100, n_jobs=50, class_weight={0: 1, 1: 1})

    def do_after_train(self, model, *args, **kwargs):
        pass


class DTModel:
    def __init__(self):
        self.name = 'DecisionTree'

    def get_new(self):
        return DecisionTreeClassifier(criterion='entropy', max_depth=10)

    def do_after_train(self, model, model_file_prefix, feature_names, class_names):
        with open(model_file_prefix + '_{}.dot'.format(self.name), 'w') as f:
            tree.export_graphviz(model, out_file=f)
            dot_data = tree.export_graphviz(model, out_file=None, feature_names=feature_names, class_names=class_names,
                                            filled=True, rounded=True, special_characters=True)
            graph = pydotplus.graph_from_dot_data(dot_data)
            graph.write_pdf(model_file_prefix + '_{}.pdf'.format(self.name))
            f = open(model_file_prefix + '_iris.png'.format(self.name), 'wb')
            f.write(graph.create_png())
            f.close()
            logging.warning('saved decision tree.')

def preprocessingGroup2(group, gid, topNum):
#	logging.warning("id "+str(gid))
	return group.tail(topNum)

def multiProcPreprocess2(dfgb, topNum, keys):
	multiprocessing.freeze_support()
	#cpus = multiprocessing.cpu_count()
	cpus = 64
	results = []
	#to test serial version
	with multiprocessing.Pool(processes=cpus) as pool: 
		for x in keys:
			result = pool.apply_async(preprocessingGroup2, args=(dfgb.get_group(x), x, topNum,))
			results.append(result)
		data = pd.concat([result.get() for result in results])
	print('multiprocess2 end one cycle')
	return data

def train_test(fault_month_start, fault_month_stop, good_month_start, good_month_stop, good_training_ratio,
               fault_training_ratio, fault_disk_take_top,
               fault_disk_drop_tail, folder, model_type):
    support_models = [RFModel(), DTModel()]
    support_models = {m.name: m for m in support_models}
    for user_model in model_type:
        if user_model not in support_models.keys():
            raise Exception('model {} not support!'.format(user_model))

    training_file_name = source_file_name_train
    testing_file_name = source_file_name_test
    if not os.path.exists(training_file_name):
        raise Exception('file {} not exists!'.format(training_file_name))
    if not os.path.exists(testing_file_name):
        raise Exception('file {} not exists!'.format(testing_file_name))

    logging.warning('reading training file:{}'.format(training_file_name))
    df_train = pd.read_csv(training_file_name,nrows = nrows_test)
    logging.warning('reading testing file:{}'.format(testing_file_name))
    df_test = pd.read_csv(testing_file_name,nrows = nrows_test)

    logging.warning('training statistics:')
    _do_statistics(df_train)
    logging.warning('testing statistics:')
    _do_statistics(df_test)

#do feature selection
    all_feature = set(df_train.columns)
    drop_feature = []
    for feature in all_feature:
        for delete_feature in delete_feature_contain:
            if len(delete_feature) <= 2:
              logging.warning('delete_feature continue :{}'.format(str(delete_feature)))  
              continue
            if delete_feature in feature:
#attention
                    #save strong feature
                    if ("SmartAttribute_Reallocated_Event_Count_RAW_VALUE" in feature) or ("SmartAttribute_Reallocated_Event_Count_VALUE" in feature):
                            if (("bin" in delete_feature_contain) & ("bin" in delete_feature)) :
                              drop_feature.append(feature)
                            elif (("diff" in delete_feature_contain) & ("diff" in delete_feature)) :
                              drop_feature.append(feature) 
                            elif (("sigma" in delete_feature_contain) & ("sigma" in delete_feature)) :
                              drop_feature.append(feature)   
                            elif (("SmartAttribute" in delete_feature_contain) & ("SmartAttribute" in delete_feature)) :
                              drop_feature.append(feature)  
                            else :
                              continue
                    else :
                        drop_feature.append(feature)
    
#    addition_save =  []
#    for single_drop in drop_feature:
#        for add_feature in additional_feature:
#            if add_feature in single_drop:
#                addition_save.append(single_drop)
#
#    additional_feature2 = ["Event_11_bin0","Event_129_bin0","Event_130_bin0","Event_140_bin0","Event_153_bin0","Event_154_bin0","Event_157_bin0","Event_158_bin0","Event_15_bin0","Event_1_bin0","Event_32_bin0","Event_34_bin0","Event_51_bin0","Event_52_bin0","Event_55_bin0","Event_7_bin0","Event_98_bin0","Event_11_bin1","Event_129_bin1","Event_130_bin1","Event_140_bin1","Event_153_bin1","Event_154_bin1","Event_157_bin1","Event_158_bin1","Event_15_bin1","Event_1_bin1","Event_32_bin1","Event_34_bin1","Event_51_bin1","Event_52_bin1","Event_55_bin1","Event_7_bin1","Event_98_bin1","Event_11_bin2","Event_129_bin2","Event_130_bin2","Event_140_bin2","Event_153_bin2","Event_154_bin2","Event_157_bin2","Event_158_bin2","Event_15_bin2","Event_1_bin2","Event_32_bin2","Event_34_bin2","Event_51_bin2","Event_52_bin2","Event_55_bin2","Event_7_bin2","Event_98_bin2","Event_11_bin3","Event_129_bin3","Event_130_bin3","Event_140_bin3","Event_153_bin3","Event_154_bin3","Event_157_bin3","Event_158_bin3","Event_15_bin3","Event_1_bin3","Event_32_bin3","Event_34_bin3","Event_51_bin3","Event_52_bin3","Event_55_bin3","Event_7_bin3","Event_98_bin3","Event_11_bin4","Event_129_bin4","Event_130_bin4","Event_140_bin4","Event_153_bin4","Event_154_bin4","Event_157_bin4","Event_158_bin4","Event_15_bin4","Event_1_bin4","Event_32_bin4","Event_34_bin4","Event_51_bin4","Event_52_bin4","Event_55_bin4","Event_7_bin4","Event_98_bin4",]
#    drop_feature  = set(drop_feature) - set(additional_feature2)
    drop_feature  = set(drop_feature) - set(additional_feature)
#    drop_feature  = set(drop_feature) - set(addition_save)
    logging.warning('drop_feature :{}'.format(str(drop_feature)))        
    
    df_train = df_train.drop(list(drop_feature), axis=1)
    df_test = df_test.drop(list(drop_feature), axis=1)
    logging.warning('after drop :{}'.format(str(set(df_train.columns))))   

# end do some feature selection

#bad get last data
#    df_train_good = df_train[df_train["LabelId"] == 0]
#    df_train_bad = df_train[df_train["LabelId"] == 1]
#    df_train_bad = multiProcPreprocess2(df_train_bad.groupby('SerialNumber'),1,df_train_bad['SerialNumber'].unique())
#    df_train = pd.concat([df_train_good,df_train_bad])
#    
#    df_test_good = df_test[df_test["LabelId"] == 0]
#    df_test_bad = df_test[df_test["LabelId"] == 1]
#    df_test_bad = multiProcPreprocess2(df_test_bad.groupby('SerialNumber'),1,df_test_bad['SerialNumber'].unique())
#    df_test = pd.concat([df_test_good,df_test_bad])    
#    logging.warning('saving test file...')
#    df_test.to_csv("test8_bad_last1.csv", index=None)
#bad get last data end


    y_train = df_train["LabelId"]
    x_train = df_train.drop(['PreciseTimeStamp', 'SerialNumber', 'LabelId','NodeId'], axis=1)#NodeId,SerialNumber,PreciseTimeStamp,
    y_test = df_test["LabelId"]
    x_test = df_test.drop(['PreciseTimeStamp', 'SerialNumber', 'LabelId','NodeId'], axis=1)

    logging.warning('final used_feature2 :{}'.format(str(set(x_train.columns))))    
    logging.warning('final used_feature count :{}'.format(str(len(x_train.columns))))  
    
## do mean shift
#    all_data = pd.concat([x_train,x_test])    
#    mean_dict = dict()
#    logging.warning('getting mean of every columns in training data...')
#    for name in x_train.columns:
#        mean_dict[name] = all_data[name].mean()
#
#    for name in x_train.columns:
#        x_train[name] = x_train[name].map(lambda x: (x - mean_dict[name]))
#        x_test[name] = x_test[name].map(lambda x: (x - mean_dict[name]))
#        logging.warning('mean shift in training and testing column {}'.format(name))
##end mean shift 
    
    
    
    for model_type_name in model_type:
        mt = support_models.get(model_type_name)
        model_file_name = experiment_type + '_' + _get_train_prefix(fault_month_start, fault_month_stop,
                                                                    good_month_start, good_month_stop,
                                                                    good_training_ratio,
                                                                    fault_training_ratio,
                                                                    fault_disk_take_top,
                                                                    fault_disk_drop_tail) + '_' + model_type_name

        model = load_model_from_file_if_exists(model_file_name)

        if model is None:
            logging.warning('training...')
            model = mt.get_new()
            model.fit(x_train, y_train)
            save_model_to_file(model, model_file_name, x_train.columns)
            logging.warning('saved model {}.'.format(model_file_name))
            mt.do_after_train(model=model, model_file_prefix=model_file_name, feature_names=x_train.columns,
                              class_names=['good disk', 'bad disk'])
        else:
            logging.warning('model exists.')
        logging.warning('testing...')
        y_hat = model.predict_proba(x_test)[:, 1]

        result_df = pd.DataFrame({
            "Scored Probabilities": y_hat,
            "LabelId": y_test
        })

        result_df.to_csv(os.path.join(folder, 'result_prob.csv'), index=None)
        draw_curve(y_hat, y_test, os.path.join(folder, model_type_name))
        logging.warning('finished {}'.format(model_type_name))
        os.chdir('..')

# generate_data(3, 10, 9, 10, 0.5, 0.5, 10, True, '1.csv')
if __name__ == "__main__":
    arguments = docopt(__doc__, version='cross_validation_by_month 1.0')

    fault_month_start = 9
    fault_month_stop = 9
    good_month_start = 9
    good_month_stop = 9
    year = 2017
    good_training_ratio = float(0.7)
    fault_training_ratio = float(0.7)
    
#  (--month=<month>) (--featureid=<featureid>)  (--delete_feature_contain=<delete_feature_contain>) (--additional_feature=<additional_feature>)   
    month = int(arguments['--month'])
    featureid = int(arguments['--featureid']) 
    delete_feature_contain = list(map(str, arguments['--delete_feature_contain'].strip().split(',')))  
    additional_feature = list(map(str, arguments['--additional_feature'].strip().split(','))) 
    
    logging.warning('month ' + str(month))
    logging.warning('featureid ' + str(featureid))    
    logging.warning('delete_feature_contain ' + str(delete_feature_contain))   
    logging.warning('additional_feature ' + str(additional_feature)) 
#    logging.warning('len additional_feature ' + str(len(additional_feature)))

#change to id1:top 5m data
#MSE_PDF_get_cv_result_small_month_9_id_1_GTR_ratio_0.3_FTR_ratio_0.3_Top_3_test
#MSE_PDF_get_cv_result_small_month_11_id_1_GTR_ratio_0.7_FTR_ratio_0.7_Top_3_train
    source_file_name_train = "MSE_PDF_get_cv_result_small_month_"+ str(month) +"_id_1_GTR_ratio_0.7_FTR_ratio_0.7_Top_3_train.csv"
    source_file_name_test = "MSE_PDF_get_cv_result_small_month_"+str(month) + "_id_1_GTR_ratio_0.3_FTR_ratio_0.3_Top_3_test.csv"
    experiment_type = 'MSE_PDF_get_cv_result_small_month_' + str(month) + "_id_" + str(featureid)

    fault_disk_take_top = int(arguments['--fault_disk_take_top'])
    source_file_name = arguments['--source_file_name']
    model_type = arguments['--model_type'].split(',')
    fault_disk_drop_tail = False if arguments['--fault_disk_drop_tail'] == 'False' else True

    folder = _get_result_folder(fault_month_start, fault_month_stop, good_month_start, good_month_stop,
                                good_training_ratio, fault_training_ratio, fault_disk_take_top,
                                fault_disk_drop_tail)

    if not os.path.exists(folder):
        logging.warning('create result folder {}.'.format(folder))
        os.makedirs(folder)
    else:
        logging.warning(
            'result folder {} exists! will overwrite files under it and appending logging file!'.format(folder))
    logging.basicConfig(level=logging.WARNING,
                        format='%(asctime)-15s %(message)s')
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

    logging.warning('generating data...')
    print(fault_month_start, fault_month_stop, good_month_start, good_month_stop, good_training_ratio,
          fault_training_ratio, fault_disk_take_top,
          fault_disk_drop_tail, source_file_name)
    logging.warning('train and testing...')
    print(fault_month_start, fault_month_stop, good_month_start, good_month_stop, good_training_ratio,
          fault_training_ratio, fault_disk_take_top,
          fault_disk_drop_tail, folder, model_type)
    train_test(fault_month_start, fault_month_stop, good_month_start, good_month_stop, good_training_ratio,
               fault_training_ratio, fault_disk_take_top,
               fault_disk_drop_tail, folder, model_type)
    
#在做其他的操作的时候可以考虑 坏盘取得最后一条信息    

