# encoding: utf-8
"""cross_validation_by_month

Usage:
    cross_validation_by_month.py
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

month_model = 9
month_data = 10
feature_id = 16
isdelete = "featureid_"+ str(feature_id) +"_result_sampleall"
experiment_type = 'test_get_predict_result_' + str(month_model) + "_predict_" + str(month_data) + "_" + isdelete

model_file_init = "test_get_cv_result_sampleall_month_"+str(month_model) +"_featureid_"+ str(feature_id) + "_GTR_ratio_0.7_FTR_ratio_0.7_Top_3_train_RandomForest"
source_file_name1 = "test_get_cv_result_sampleall_month_" + str(month_data) + "_featureid_3_GTR_ratio_0.7_FTR_ratio_0.7_Top_3_train.csv"
source_file_name2 = "test_get_cv_result_sampleall_month_" + str(month_data) + "_featureid_3_GTR_ratio_0.3_FTR_ratio_0.3_Top_3_test.csv"

delete_feature_contain =[
     "Event_1_bin",
    "Event_7_bin",
    "Event_15_bin",
    "Event_32_bin",
    "Event_34_bin",
    "Event_51_bin",
    "Event_52_bin",
    "Event_55_bin",
    "Event_98_bin",
    "Event_129_bin",
    "Event_130_bin",
    "Event_140_bin",
    "Event_153_bin",
    "Event_154_bin",
    "Event_157_bin",
    "Event_158_bin",
    
    "Event_1_diff",
    "Event_7_diff",
    "Event_11_diff",
    "Event_15_diff",
    "Event_32_diff",
    "Event_34_diff",
    "Event_51_diff",
    "Event_52_diff",
    "Event_55_diff",
    "Event_98_diff",
    "Event_129_diff",
    "Event_130_diff",
    "Event_140_diff",
    "Event_153_diff",
    "Event_154_diff",
    "Event_157_diff",
    "Event_158_diff",
 
    "Event_1_sigma",
    "Event_7_sigma",
    "Event_11_sigma",
    "Event_15_sigma",
    "Event_32_sigma",
    "Event_34_sigma",
    "Event_51_sigma",
    "Event_52_sigma",
    "Event_55_sigma",
    "Event_98_sigma",
    "Event_129_sigma",
    "Event_130_sigma",
    "Event_140_sigma",
    "Event_153_sigma",
    "Event_154_sigma",
    "Event_157_sigma",
    "Event_158_sigma",
    "Event",
    "diff",
    "sigma"
]

additional_feature = [
    "Event_15_sigma3",
#    "Event_129_sigma3",
#    "Event_129_diff4",
#    "Event_153_sigma3",
#    "Event_153_diff4",
"SmartAttribute_Reallocated_Event_Count_RAW_VALUE",
"SmartAttribute_Reallocated_Event_Count_VALUE",
#"Event_153",
        ]
year= None
addsum = False
adddiff = False
addsigma = False
topnum = 3
nrows_test = None
header_file_name = 'header.csv'

def _get_experiment_base(fault_month_start, fault_month_stop, good_month_start, good_month_stop, good_training_ratio,
                         fault_training_ratio, fault_disk_take_top,
                         fault_disk_drop_tail):
    return 'GTR_ratio_{}_FTR_ratio_{}_F_top_{}'.format(round(good_training_ratio, 2),round(fault_training_ratio, 2), topnum)


def _get_train_prefix(fault_month_start, fault_month_stop, good_month_start, good_month_stop, good_training_ratio,
                      fault_training_ratio, fault_disk_take_top,
                      fault_disk_drop_tail):
    return _get_experiment_base(fault_month_start, fault_month_stop, good_month_start, good_month_stop,
                                good_training_ratio, fault_training_ratio,
                                fault_disk_take_top, fault_disk_drop_tail) + '_train'


def _get_test_prefix(fault_month_start, fault_month_stop, good_month_start, good_month_stop, good_training_ratio,
                     fault_training_ratio, fault_disk_take_top,
                     fault_disk_drop_tail):
    return _get_experiment_base(fault_month_start, fault_month_stop, good_month_start, good_month_stop,
                                1 - good_training_ratio, 1 - fault_training_ratio,
                                fault_disk_take_top, fault_disk_drop_tail) + '_test'


def _get_train_file_name(fault_month_start, fault_month_stop, good_month_start, good_month_stop, good_training_ratio,
                         fault_training_ratio, fault_disk_take_top,
                         fault_disk_drop_tail):
    return experiment_type + '_' + _get_train_prefix(fault_month_start, fault_month_stop, good_month_start,
                                                     good_month_stop, good_training_ratio, fault_training_ratio,
                                                     fault_disk_take_top, fault_disk_drop_tail) + '.csv'


def _get_test_file_name(fault_month_start, fault_month_stop, good_month_start, good_month_stop, good_training_ratio,
                        fault_training_ratio, fault_disk_take_top,
                        fault_disk_drop_tail):
    return experiment_type + '_' + _get_test_prefix(fault_month_start, fault_month_stop, good_month_start,
                                                    good_month_stop, good_training_ratio, fault_training_ratio,
                                                    fault_disk_take_top, fault_disk_drop_tail) + '.csv'


def _get_result_folder(fault_month_start, fault_month_stop, good_month_start, good_month_stop, good_training_ratio,
                       fault_training_ratio, fault_disk_take_top,
                       fault_disk_drop_tail):
    return experiment_type + '_' + _get_experiment_base(fault_month_start, fault_month_stop, good_month_start,
                                                        good_month_stop, good_training_ratio,
                                                        fault_training_ratio,
                                                        fault_disk_take_top, fault_disk_drop_tail)
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
        return RandomForestClassifier(n_estimators=100, n_jobs=50, class_weight={0: 1, 1: 1000})

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


def train_test(fault_month_start, fault_month_stop, good_month_start, good_month_stop, good_training_ratio,
               fault_training_ratio, fault_disk_take_top,
               fault_disk_drop_tail, folder, model_type):
    support_models = [RFModel(), DTModel()]
    support_models = {m.name: m for m in support_models}
    for user_model in model_type:
        if user_model not in support_models.keys():
            raise Exception('model {} not support!'.format(user_model))

    training_file_name = source_file_name1
    testing_file_name = source_file_name2

    if not os.path.exists(training_file_name):
        raise Exception('file {} not exists!'.format(training_file_name))
    if not os.path.exists(testing_file_name):
        raise Exception('file {} not exists!'.format(testing_file_name))

    logging.warning('reading training file:{}'.format(training_file_name))
    df_train = pd.read_csv(training_file_name)
    logging.warning('reading testing file:{}'.format(testing_file_name))
    df_test = pd.read_csv(testing_file_name)
    
    df_test = pd.concat([df_train,df_test])

    logging.warning('testing statistics:')
    _do_statistics(df_test)


#do feature selection

    all_feature = set(df_test.columns)
    drop_feature = []
    for feature in all_feature:
        for delete_feature in delete_feature_contain:
            if delete_feature in feature:
                drop_feature.append(feature)
                
    drop_feature  = set(drop_feature) - set(additional_feature)    
    logging.warning('drop_feature :{}'.format(str(drop_feature)))        
    df_test = df_test.drop(list(drop_feature), axis=1)

#do some feature selection

    y_test = df_test["LabelId"]
    x_test = df_test.drop(['PreciseTimeStamp', 'SerialNumber', 'LabelId', 'NodeId'], axis=1)

    logging.warning('final used_feature2 :{}'.format(str(set(x_test.columns))))    
    logging.warning('final used_feature count :{}'.format(str(len(x_test.columns))))  


    for model_type_name in model_type:
        model = load_model_from_file_if_exists(model_file_init)
        if model is None:
#            logging.warning('training...')
#            model = mt.get_new()
#            model.fit(x_train, y_train)
#            save_model_to_file(model, model_file_name, x_train.columns)
#            logging.warning('saved model {}.'.format(model_file_name))
#            mt.do_after_train(model=model, model_file_prefix=model_file_name, feature_names=x_train.columns,
#                              class_names=['good disk', 'bad disk'])
            logging.warning('big bug.')
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
#    generate_data(fault_month_start, fault_month_stop, good_month_start, good_month_stop, good_training_ratio,
#                  fault_training_ratio, fault_disk_take_top,
#                  fault_disk_drop_tail, source_file_name)
#
#    logging.warning('train and testing...')
    print(fault_month_start, fault_month_stop, good_month_start, good_month_stop, good_training_ratio,
          fault_training_ratio, fault_disk_take_top,
          fault_disk_drop_tail, folder, model_type)
    train_test(fault_month_start, fault_month_stop, good_month_start, good_month_stop, good_training_ratio,
               fault_training_ratio, fault_disk_take_top,
               fault_disk_drop_tail, folder, model_type)
    
# --good_training_ratio=0.7   --fault_training_ratio=0.7