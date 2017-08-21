# source: https://github.com/Azure/fast_retraining/blob/master/experiments/04_PlanetKaggle.ipynb
# source: https://github.com/Azure/fast_retraining/blob/master/experiments/04_PlanetKaggle_GPU.ipynb
# source: https://github.com/Azure/fast_retraining/blob/master/experiments/libs/loaders.py
# source: https://github.com/Azure/fast_retraining/blob/master/experiments/libs/planet_kaggle.py
# source: https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/data/train_v2.csv.zip

from __future__ import print_function

import glob
import logging
import os
import shutil
import sys

from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.resnet50 import ResNet50
import lightgbm as lgbm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import xgboost as xgb

from metrics import *
from utils import *


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def labels_from(labels_df):
    """ Extracts the unique labels from the labels dataframe
    """
    # Build list with unique labels
    label_list = []
    for tag_str in labels_df.tags.values:
        labels = tag_str.split(' ')
        for label in labels:
            if label not in label_list:
                label_list.append(label)
    return label_list


def enrich_with_feature_encoding(labels_df):
    # Add onehot features for every label
    for label in labels_from(labels_df):
        labels_df[label] = labels_df['tags'].apply(lambda x: 1 if label in x.split(' ') else 0)
    return labels_df


def to_multi_label_dict(enriched_labels_df):
    df = enriched_labels_df.set_index('image_name').drop('tags', axis=1)
    return dict((filename, encoded_array) for filename, encoded_array in zip(df.index, df.values))


def get_file_count(folderpath):
    """ Returns the number of files in a folder
    """
    return len(glob.glob(folderpath))


def threshold_prediction(pred_y, threshold=0.5):# TODO: Needs to be tuned?
    return pred_y > threshold


def read_images(filepath, filenames):
    """ Read images in batches
    """
    img_data = list()
    for name in filenames:
        img_path = os.path.join(filepath, name+'.jpg')
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        img_data.append(preprocess_input(x))
    return np.concatenate(img_data)


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def featurise_images(model, filepath, nameformat, num_iter, batch_size=32, desc=None):
    """ Use DL model to featurise images
    """
    features = list()
    img_names = list()
    num_list = list(num_iter)
    num_batches = np.ceil(len(num_list)/batch_size)
    
    for num_chunk in tqdm(chunks(num_list, batch_size), total=num_batches, desc=desc):
        filenames = [nameformat.format(index) for index in num_chunk]
        batch_images = read_images(filepath, filenames)
        img_names.extend(filenames)
        features.extend(model.predict_on_batch(batch_images).squeeze())
    return np.array(features), img_names
  
    
def generate_validation_files(train_path, val_path, num_train = 35000):
    """ Creates the validation files from the train files.
    """
    num_train_ini = get_file_count(os.path.join(train_path, '*.jpg'))
    assert num_train_ini > num_train
    
    order = ('mv ' + train_path + '/train_{' + str(num_train) + '..' + str(num_train_ini-1) +
             '}.jpg ' + val_path)
    os.system(order)
    

def load_planet(db_folder, full_csv_path):
    csv_path = os.path.join(db_folder, 'train_v2.csv')
    train_path = os.path.join(db_folder, 'train-jpg')
    val_path = os.path.join(db_folder, 'validate-jpg')
    assert os.path.isfile(csv_path)
    assert os.path.exists(train_path)
    if not os.path.exists(val_path): os.mkdir(val_path)
    if not os.listdir(val_path): 
        logger.info('Validation folder is empty, moving files...')
        generate_validation_files(train_path, val_path)
    
    logger.info('Reading in labels')
    #labels_df = pd.read_csv(csv_path).pipe(enrich_with_feature_encoding)
    labels_df = pd.read_csv(csv_path).pipe(enrich_with_feature_encoding)
    multi_label_dict = to_multi_label_dict(labels_df)
    
    nb_train_samples = get_file_count(os.path.join(train_path, '*.jpg'))
    nb_validation_samples = get_file_count(os.path.join(val_path, '*.jpg'))
    nb_train_samples_base = get_file_count(os.path.join(train_path, '*.jpg'))
    #nb_train_samples = 100
    #nb_validation_samples = 20

    logger.debug('Number of training files {}'.format(nb_train_samples))
    logger.debug('Number of validation files {}'.format(nb_validation_samples))
    logger.debug('Loading model')

    model = ResNet50(include_top=False)
    train_features, train_names = featurise_images(
        model, train_path, 'train_{}', range(nb_train_samples),
        desc='Featurising training images')

    validation_features, validation_names = featurise_images(
        model, val_path, 'train_{}',
        range(nb_train_samples_base, nb_train_samples_base+nb_validation_samples),
        desc='Featurising validation images')

    print(train_features.shape)
    print(validation_features.shape)

    # Prepare data
    y_train = np.array([multi_label_dict[name] for name in train_names])
    y_val = np.array([multi_label_dict[name] for name in validation_names])

    save_full(train_names, train_features, y_train,
              validation_names, validation_features, y_val,
              full_csv_path)

    return train_features, validation_features, y_train, y_val


def load_full(path):
    n_features = 2048
    n_labels = 17
    # load the data in float32
    dtype = np.float32
    #dtype_dict = {'name' : 'string' }
    dtype_dict = {}
    dtype_dict.update({'feature_%d' % i : dtype for i in range(n_features)})
    dtype_dict.update({'label_%d' % i : dtype for i in range(n_labels)})
    df = pd.read_csv(path, sep=' ', dtype=dtype_dict, index_col='name')
    n_train = 35000
    val_base = n_train
    n_val = 5479
    df_train = df.loc[['train_%d' % i for i in range(n_train)], :]
    df_val = df.loc[['train_%d' % (val_base + i) for i in range(n_val)], :]
    feature_cols = ['feature_%d' % i for i in range(n_features)]
    label_cols = ['label_%d' % i for i in range(n_labels)]
    X_train = df_train.loc[:, feature_cols].values
    X_test = df_val.loc[:, feature_cols].values
    y_train = df_train.loc[:, label_cols].values
    y_test = df_val.loc[:, label_cols].values
    return X_train, X_test, y_train, y_test


def save_full(
        train_names, train_features, train_labels,
        val_names, val_features, val_labels, path):
    names = train_names + val_names
    features = np.concatenate((train_features, val_features), axis=0)
    labels = np.concatenate((train_labels, val_labels), axis=0)
    n_features = features.shape[1]
    n_labels = labels.shape[1]
    print('n_features=%d, n_labels=%d' % (n_features, n_labels))
    feature_cols = {'feature_%d' % i for i in range(n_features)}
    label_cols = {'label_%d' % i for i in range(n_labels)}
    cols = {'name': names}
    cols.update({'feature_%d' % i : features[:, i] for i in range(n_features)})
    cols.update({'label_%d' % i : labels[:, i] for i in range(n_labels)})
    df = pd.DataFrame.from_dict(cols, orient='columns')

    cols_to_save = (['name'] + ['feature_%d' % i for i in range(n_features)] +
      ['label_%d' % i for i in range(n_labels)])
    df.to_csv(path, sep=' ', index=False, columns=cols_to_save)
    print(df)


def prepare(db_folder):
    full_csv_name = 'train_full.csv'
    full_csv_path = os.path.join(db_folder, full_csv_name)

    start = time.time()
    if not os.path.isfile(full_csv_path):
        # preprocess data, ignore the returned result
        load_planet(db_folder, full_csv_path)
        
    X_train, X_test, y_train, y_test = load_full(full_csv_path)
    load_time = time.time() - start
    print('Planet dataset loaded in %.2fs' % load_time, file=sys.stderr)
    return Data(X_train, X_test, y_train, y_test)


planet_num_rounds = 50


class XgbParamsPlanet(XgbGpuBenchmark):

    num_rounds = planet_num_rounds

    # do not report accuracy
    def accuracy(self):
        return {}


class LgbmParamsPlanet(LgbmGpuBenchmark):

    num_rounds = planet_num_rounds

    # do not report accuracy
    def accuracy(self):
        return {}


# invididual models are trained per-class in this case;
# therefore, a different code is required to run the benchmark
class PlanetBenchmark(Benchmark):

    benchmark_cls = None

    def run(self):
        n_classes = self.y_train.shape[1]

        # uncomment this to train fewer classes, and run the experiment faster
        #n_classes = 3
        #self.y_test = self.y_test[:, range(n_classes)]
        
        n_test = self.y_test.shape[0]
        self.y_prob = np.zeros((n_test, n_classes))

        results = {'train_time': 0, 'test_time': 0}

        for i_class in range(n_classes):
            # data for the per-class benchmark
            y_train_class = self.y_train[:, i_class]
            y_test_class = self.y_test[:, i_class]
            data_class = Data(self.X_train, self.X_test, y_train_class, y_test_class)

            # run per-class benchmark
            benchmark_class = self.benchmark_cls(data_class, self.params)
            results_class = benchmark_class.run()

            # copy the prediction data
            self.y_prob[:, i_class] = benchmark_class.y_prob
            
            # aggregate statistics
            results['train_time'] += results_class['train_time']
            results['test_time'] += results_class['test_time']

        # compute overall accuracy metrics
        y_pred = self.y_prob > 0.1
        results['accuracy'] = classification_metrics_average(
            self.y_test, y_pred, average='samples')

        return results


class XgbPlanetBenchmark(PlanetBenchmark):

    benchmark_cls = XgbParamsPlanet


class LgbmPlanetBenchmark(PlanetBenchmark):

    benchmark_cls = LgbmParamsPlanet


xgb_cpu_params = {
    'max_depth':6,
    'objective':'binary:logistic',
    'min_child_weight':1,
    'learning_rate':0.1,
    'scale_pos_weight':2,
    'gamma':0.1,
    'reg_lamda':1,
    'subsample':1,
    'nthread':get_number_processors(),
}

xgb_cpu_hist_params = {
    'max_depth':0,
    'max_leaves':2**6,
    'objective':'binary:logistic',
    'min_child_weight':1,
    'learning_rate':0.1,
    'scale_pos_weight':2,
    'gamma':0.1,
    'reg_lamda':1,
    'subsample':1,
    'nthread':get_number_processors(),
    'tree_method':'hist',
    'grow_policy':'lossguide',
    'max_bins': 63,
}

lgbm_cpu_params = {
    'num_leaves': 2**6,
    'learning_rate': 0.1,
    'scale_pos_weight': 2,
    'min_split_gain': 0.1,
    'min_child_weight': 1,
    'reg_lambda': 1,
    'subsample': 1,
    'objective':'binary',
    'task': 'train',
    'nthread':get_number_processors(),
    'max_bin': 63,
}

xgb_gpu_params = {
    'max_depth':2,
    'objective':'binary:logistic',
    'min_child_weight':1,
    'learning_rate':0.1,
    'scale_pos_weight':2,
    'gamma':0.1,
    'reg_lamda':1,
    'subsample':1,
    'tree_method':'exact',
    'updater':'grow_gpu',
}

xgb_gpu_hist_params = {
    'max_depth':7,
    'max_leaves':2**6,
    'objective':'binary:logistic',
    'min_child_weight':1,
    'learning_rate':0.1,
    'scale_pos_weight':2,
    'gamma':0.1,
    'reg_lamda':1,
    'subsample':1,
    'tree_method':'gpu_hist',
    'max_bins': 63,
}

lgbm_gpu_params = {
    'num_leaves': 2**6,
    'learning_rate': 0.1,
    'scale_pos_weight': 2,
    'min_split_gain': 0.1,
    'min_child_weight': 1,
    'reg_lambda': 1,
    'subsample': 1,
    'objective':'binary',
    'device': 'gpu',
    'task': 'train',
    'max_bin': 63,
}

benchmarks = {
    'xgb-cpu': (XgbPlanetBenchmark, xgb_cpu_params),
    'xgb-cpu-hist': (XgbPlanetBenchmark, xgb_cpu_hist_params),
    'lgbm-cpu': (LgbmPlanetBenchmark, lgbm_cpu_params),
    # 'xgb-gpu' runs out of memory
    #'xgb-gpu': (XgbPlanetBenchmark, xgb_gpu_params),
    'xgb-gpu-hist': (XgbPlanetBenchmark, xgb_gpu_hist_params),
    # 'lgbm-gpu' runs out of host memory with all 17 classes
    # TODO: look into this
    #'lgbm-gpu': (LgbmPlanetBenchmark, lgbm_gpu_params),
}
