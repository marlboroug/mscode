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
month = 11
featureid = 1
keep_feature =[]
source_file_name_good = "small_new_feature_" + str(month) + "month_feature_Good.csv"#test_8_month_feature_bad_python_all  test_8_month_feature_Good_python_all
source_file_name_bad = "all_bad" + str(month) + "_features.csv"
experiment_type = 'MSE_PDF_get_cv_result_small_month_' + str(month) + "_id_" + str(featureid)

topnum = 3
nrows_test = None

folder = ""

def _get_experiment_base(fault_month_start, fault_month_stop, good_month_start, good_month_stop, good_training_ratio,
                         fault_training_ratio, fault_disk_take_top,
                         fault_disk_drop_tail):
    return 'GTR_ratio_{}_FTR_ratio_{}_Top_{}'.format(round(good_training_ratio, 2),round(fault_training_ratio, 2), topnum)


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

def generate_data(fault_month_start, fault_month_stop, good_month_start, good_month_stop, good_training_ratio,
                  fault_training_ratio, fault_disk_take_top,
                  fault_disk_drop_tail, source_file_name):
    training_file_name = _get_train_file_name(fault_month_start, fault_month_stop, good_month_start, good_month_stop,
                                              good_training_ratio, fault_training_ratio,
                                              fault_disk_take_top,
                                              fault_disk_drop_tail)
    testing_file_name = _get_test_file_name(fault_month_start, fault_month_stop, good_month_start, good_month_stop,
                                            good_training_ratio, fault_training_ratio,
                                            fault_disk_take_top,
                                            fault_disk_drop_tail)

    logging.warning('reading source file...')
    logging.warning('reading source file...' + source_file_name_good)
    good = pd.read_csv(source_file_name_good,nrows = nrows_test)
    
    logging.warning('reading source file...' + source_file_name_bad)
    bad =  pd.read_csv(source_file_name_bad, nrows = nrows_test)
    
#    to_be_deleted = []
#    for name in bad.columns:
#        if name in ['model', 'capacity_bytes','FaultTimeStamp','EventId','ParsingError','AAMFeature','APMFeature','ATASecurity','ATAVersion','Device','DeviceModel','FirmwareVersion','ModelFamily','RdLookAhead','RotationRate','SMARTSupport','SATAVersion','SectorSize','UserCapacity','WriteCache','WtCacheReorder']:
#            to_be_deleted.append(name)
#        elif name == "LabelId" or name == "SerialNumber" or name == 'PreciseTimeStamp' or  name == 'NodeId':
#            continue
#        elif  str(bad[name].std()) == "nan": #bad8[name].mean == 0 or
#            to_be_deleted.append(name)
#  
#    bad = bad.drop(to_be_deleted, axis=1)
#    #delete bad data
    

    fault_disks = bad['SerialNumber'].unique()
    fault_disks = list(fault_disks)
    logging.warning('all fault disks:{}'.format(len(fault_disks)))

    random.shuffle(fault_disks)
    fault_train_disks, fault_test_disks = fault_disks[0: int(len(fault_disks) * fault_training_ratio)], fault_disks[int(len(fault_disks) * fault_training_ratio):]
    logging.warning('after ratio split, training fault disks:{}, testing fault disks:{}'.format(len(fault_train_disks),len(fault_test_disks)))

    good_disks = good['SerialNumber'].unique()
    logging.warning('all good disks1:{}'.format(len(good_disks)))
    good_disks = list(set(good_disks) - set(fault_disks))
    logging.warning('all good disks2:{}'.format(len(good_disks)))

    random.shuffle(good_disks)
    good_train_disks, good_test_disks = good_disks[0: int(len(good_disks) * good_training_ratio)], good_disks[int(len(good_disks) * good_training_ratio):]
    logging.warning('after ratio split, training good disks:{}, testing good disks:{}'.format(len(good_train_disks),len(good_test_disks)))
    

# get top number 3    
#    good = multiProcPreprocess2(good.groupby('SerialNumber'),3,set(good_disks)) 
#    bad = multiProcPreprocess2(bad.groupby('SerialNumber'),3,set(fault_disks)) 

#    good_mean_dict = dict()
#    bad_mean_dict = dict()
#    for name in good.columns:
#        if name in ['NodeId', 'SerialNumber', 'PreciseTimeStamp','LabelId']:
#              continue
#        good_mean_dict[name] = good[name].mean()
#        bad_mean_dict[name] = bad[name].mean()
#        
#    for name in good.columns:
#        if name in ['NodeId', 'SerialNumber', 'PreciseTimeStamp','LabelId']:
#              continue
#        good[name][np.isnan(good[name])] = good_mean_dict[name]
#        bad[name][np.isnan(bad[name])] = bad_mean_dict[name]
#        logging.warning('done replace nan with mean in data and testing column {}'.format(name))
        
    
    logging.warning('filtering good training disks data...')
    good_train_df = good[(good['SerialNumber'].isin(good_train_disks)) ]

    logging.warning('filtering fault training disks data...')
    fault_train_df = bad[(bad['SerialNumber'].isin(fault_train_disks))]

    logging.warning('filtering good testing disks data...')
    good_test_df = good[(good['SerialNumber'].isin(good_test_disks))]

    logging.warning('filtering fault testing disks data...')
    fault_test_df = bad[(bad['SerialNumber'].isin(fault_test_disks)) ]
#
## add feature enginging
#    good_train_df = addColumns(good_train_df)
#    logging.warning('feature engining good training disks data...')
#    good_train_df = multiProcPreprocess(good_train_df.groupby('SerialNumber'),topnum,good_train_disks)#def multiProcPreprocess(dfgb, topNum, keys):
#    logging.warning('after multiProcPreprocess good training disks data...' + str(len(good_train_df)))
#
#    fault_train_df = addColumns(fault_train_df)        
#    logging.warning('feature engining fault training disks data...')
#    fault_train_df = multiProcPreprocess(fault_train_df.groupby('SerialNumber'),topnum,fault_train_disks)
#    logging.warning('after multiProcPreprocess good training disks data...' + str(len(fault_train_df)))
#    
#    good_test_df = addColumns(good_test_df)  
#    logging.warning('feature engining :' + str(good_test_df.columns))    
#    logging.warning('feature engining  good testing disks data...')
#    good_test_df =  multiProcPreprocess(good_test_df.groupby('SerialNumber'),topnum,good_test_disks)
#    logging.warning('after multiProcPreprocess good training disks data...' + str(len(good_test_df)))
#
#    fault_test_df = addColumns(fault_test_df)   
#    logging.warning('feature engining  fault testing disks data...')
#    fault_test_df = multiProcPreprocess(fault_test_df.groupby('SerialNumber'),topnum,fault_test_disks)
#    logging.warning('after multiProcPreprocess good training disks data...' + str(len(fault_test_df)))
## done

    logging.warning('group by disk id and then sort and take fault training data...')
    train_df = pd.concat([good_train_df,fault_train_df])
    
    logging.warning('saving training file...')
    train_df.to_csv(training_file_name, index=None)
    logging.warning('saved training file.')

    logging.warning('group by disk id and then sort and take fault testing data...')
    test_df = pd.concat([good_test_df,fault_test_df])  
    
    logging.warning('saving testing file...')
    test_df.to_csv(testing_file_name, index=None)
    logging.warning('saved testing file.')
def save_model_to_file(model, model_file_prefix, columns):
    with open(folder + "/" + model_file_prefix + '.model', 'wb') as f:
        cPickle.dump(model, f)
        importance = model.feature_importances_
        importance = pd.DataFrame(importance, index=columns,
                                  columns=["Importance"])

        importance["Std"] = np.std([tree.feature_importances_
                                    for tree in model.estimators_], axis=0)
        importance.to_csv('{}/{}_feature_importance.csv'.format(folder,model_file_prefix))


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

    training_file_name = _get_train_file_name(fault_month_start, fault_month_stop, good_month_start, good_month_stop,
                                              good_training_ratio, fault_training_ratio,
                                              fault_disk_take_top,
                                              fault_disk_drop_tail)
    testing_file_name = _get_test_file_name(fault_month_start, fault_month_stop, good_month_start, good_month_stop,
                                            good_training_ratio, fault_training_ratio,
                                            fault_disk_take_top,
                                            fault_disk_drop_tail)

    if not os.path.exists(training_file_name):
        raise Exception('file {} not exists!'.format(training_file_name))
    if not os.path.exists(testing_file_name):
        raise Exception('file {} not exists!'.format(testing_file_name))

    logging.warning('reading training file:{}'.format(training_file_name))
    df_train = pd.read_csv(training_file_name)
    logging.warning('reading testing file:{}'.format(testing_file_name))
    df_test = pd.read_csv(testing_file_name)

    logging.warning('training statistics:')
    _do_statistics(df_train)
    logging.warning('testing statistics:')
    _do_statistics(df_test)

    y_train = df_train["LabelId"]
    x_train = df_train.drop(['PreciseTimeStamp', 'SerialNumber', 'LabelId','NodeId'], axis=1)#NodeId,SerialNumber,PreciseTimeStamp,
    y_test = df_test["LabelId"]
    x_test = df_test.drop(['PreciseTimeStamp', 'SerialNumber', 'LabelId','NodeId'], axis=1)

#    train_mean_dict = dict()
#    test_mean_dict = dict()
#    logging.warning('getting mean of every columns in training data...')
#    for name in x_train.columns:
#        train_mean_dict[name] = x_train[name].mean()
#        test_mean_dict[name] = x_test[name].mean()
#
#    for name in x_train.columns:
#        x_train[name][np.isnan(x_train[name])] = train_mean_dict[name]
#        x_test[name][np.isnan(x_test[name])] = test_mean_dict[name]
#        logging.warning('done replace nan with mean in training and testing column {}'.format(name))

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
    generate_data(fault_month_start, fault_month_stop, good_month_start, good_month_stop, good_training_ratio,
                  fault_training_ratio, fault_disk_take_top,
                  fault_disk_drop_tail, source_file_name)

    logging.warning('train and testing...')
    print(fault_month_start, fault_month_stop, good_month_start, good_month_stop, good_training_ratio,
          fault_training_ratio, fault_disk_take_top,
          fault_disk_drop_tail, folder, model_type)
    train_test(fault_month_start, fault_month_stop, good_month_start, good_month_stop, good_training_ratio,
               fault_training_ratio, fault_disk_take_top,
               fault_disk_drop_tail, folder, model_type)
    
  