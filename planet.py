# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import print_function

import glob
import os
import shutil
import sys

from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.resnet50 import ResNet50
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from metrics import *
from utils import *


def labels_from(labels_df):
    """Extracts the unique labels from the labels dataframe"""
    # Build list with unique labels
    label_list = []
    for tag_str in labels_df.tags.values:
        labels = tag_str.split(" ")
        for label in labels:
            if label not in label_list:
                label_list.append(label)
    return label_list

def enrich_with_feature_encoding(labels_df):
    # Add onehot features for every label
    for label in labels_from(labels_df):
        labels_df[label] = labels_df["tags"].apply(lambda x: 1 if label in x.split(" ") else 0)
    return labels_df

def to_multi_label_dict(enriched_labels_df):
    df = enriched_labels_df.set_index("image_name").drop("tags", axis=1)
    return dict((filename, encoded_array) for filename, encoded_array in zip(df.index, df.values))

def get_file_count(folderpath):
    """Returns the number of files in a folder"""
    return len(glob.glob(folderpath))

def threshold_prediction(pred_y, threshold=0.5):# TODO: Needs to be tuned?
    return pred_y > threshold

def read_images(filepath, filenames):
    """Read images in batches"""
    img_data = list()
    for name in filenames:
        img_path = os.path.join(filepath, name+".jpg")
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        img_data.append(preprocess_input(x))
    return np.concatenate(img_data)

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def featurise_images(model, filepath, nameformat, num_iter, batch_size=32, desc=None):
    """Use DL model to featurise images"""
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
    """Creates the validation files from the train files."""
    num_train_ini = get_file_count(os.path.join(train_path, "*.jpg"))
    assert num_train_ini > num_train
    for i in range(num_train, num_train_ini):
        subprocess.check_call("mv %s/train_%d.jpg %s" % (train_path, i, val_path), shell=True)

def save_processed(train_names, train_features, train_labels,
                   val_names, val_features, val_labels, path):
    names = train_names + val_names
    features = np.concatenate((train_features, val_features), axis=0)
    labels = np.concatenate((train_labels, val_labels), axis=0)
    n_features = features.shape[1]
    n_labels = labels.shape[1]
    print("n_features=%d, n_labels=%d" % (n_features, n_labels))
    feature_cols = {"feature_%d" % i for i in range(n_features)}
    label_cols = {"label_%d" % i for i in range(n_labels)}
    cols = {"name": names}
    cols.update({"feature_%d" % i : features[:, i] for i in range(n_features)})
    cols.update({"label_%d" % i : labels[:, i] for i in range(n_labels)})
    df = pd.DataFrame.from_dict(cols, orient="columns")
    cols_to_save = (["name"] + ["feature_%d" % i for i in range(n_features)] +
      ["label_%d" % i for i in range(n_labels)])
    df.to_csv(path, sep=" ", index=False, columns=cols_to_save)

def load_planet(db_folder, full_csv_path):
    zip_csv = "train_v2.csv.zip"
    csv_path = os.path.join(db_folder, "train_v2.csv")
    if not os.path.exists(csv_path):
        print("Unzipping csv...")
        subprocess.check_call("cd %s && unzip -o %s" % (db_folder, zip_csv), shell=True)
    train_path = os.path.join(db_folder, "train-jpg")
    if not os.path.exists(train_path):
        print("Unzipping images...")
        subprocess.check_call("cd %s && tar xf train-jpg.tar" % db_folder, shell=True)
    val_path = os.path.join(db_folder, "validate-jpg")
    if not os.path.exists(val_path):
        os.mkdir(val_path)
    if not os.listdir(val_path):
        print("Validation folder is empty, moving files...")
        generate_validation_files(train_path, val_path)    
    print("Reading in labels")
    labels_df = pd.read_csv(csv_path).pipe(enrich_with_feature_encoding)
    multi_label_dict = to_multi_label_dict(labels_df)
    nb_train_samples = get_file_count(os.path.join(train_path, "*.jpg"))
    nb_validation_samples = get_file_count(os.path.join(val_path, "*.jpg"))
    nb_train_samples_base = get_file_count(os.path.join(train_path, "*.jpg"))
    print("Number of training files {}".format(nb_train_samples))
    print("Number of validation files {}".format(nb_validation_samples))
    print("Loading model")
    model = ResNet50(include_top=False)
    train_features, train_names = featurise_images(
        model, train_path, "train_{}", range(nb_train_samples),
        desc="Featurising training images")
    validation_features, validation_names = featurise_images(
        model, val_path, "train_{}",
        range(nb_train_samples_base, nb_train_samples_base+nb_validation_samples),
        desc="Featurising validation images")
    print("Train data: ", train_features.shape)
    print("Validation data: ", validation_features.shape)
    # Prepare data
    y_train = np.array([multi_label_dict[name] for name in train_names])
    y_val = np.array([multi_label_dict[name] for name in validation_names])
    save_processed(train_names, train_features, y_train,
                   validation_names, validation_features, y_val,
                   full_csv_path)

def load_processed(path):
    print("Loading processed dataset...")
    n_features = 2048
    n_labels = 17
    # load the data in float32
    dtype = np.float32
    dtype_dict = {}
    dtype_dict.update({"feature_%d" % i : dtype for i in range(n_features)})
    dtype_dict.update({"label_%d" % i : dtype for i in range(n_labels)})
    df = pd.read_csv(path, sep=" ", dtype=dtype_dict, index_col="name")
    n_train = 35000
    val_base = n_train
    n_val = 5479
    df_train = df.loc[["train_%d" % i for i in range(n_train)], :]
    df_val = df.loc[["train_%d" % (val_base + i) for i in range(n_val)], :]
    feature_cols = ["feature_%d" % i for i in range(n_features)]
    label_cols = ["label_%d" % i for i in range(n_labels)]
    X_train = df_train.loc[:, feature_cols].values
    X_test = df_val.loc[:, feature_cols].values
    y_train = df_train.loc[:, label_cols].values
    y_test = df_val.loc[:, label_cols].values
    return X_train, X_test, y_train, y_test

# TODO: support for prepareImpl, like other datasets

def prepare(dbFolder, nrows):
    full_csv_name = "train_full.csv"
    full_csv_path = os.path.join(dbFolder, full_csv_name)
    start = time.time()
    if not os.path.exists(full_csv_path):
        load_planet(dbFolder, full_csv_path)        
    X_train, X_test, y_train, y_test = load_processed(full_csv_path)
    load_time = time.time() - start
    print("Planet dataset loaded in %.2fs" % load_time)
    return Data(X_train, X_test, y_train, y_test)

nthreads = get_number_processors()
nTrees = 50

def metrics(y_test, y_prob):
    y_pred = y_prob > 0.1
    return classification_metrics_average(y_test, y_pred, avg="samples")

def catLabels(y_prob):
    return np.argmax(y_prob, axis=1)

# invididual models are trained per-class in this case;
# therefore, a different code is required to run the benchmark
class PlanetBenchmark(Benchmark):
    def prepare(self):
        self.benchmark_cls = self.params["benchmark_cls"]
        del self.params["benchmark_cls"]
        if "postprocess" in self.params:
            self.postprocess = self.params["postprocess"]
            del self.params["postprocess"]
        else:
            self.postprocess = None
        Benchmark.prepare(self)

    def run(self):
        self.prepare()
        n_classes = self.data.y_train.shape[1]
        n_test = self.data.y_test.shape[0]
        self.y_pred = np.zeros((n_test, n_classes))
        train_time = test_time = 0
        for i_class in range(n_classes):
            print("benchmarking class %d" % i_class)
            # data for the per-class benchmark
            y_train_class = self.data.y_train[:, i_class]
            y_test_class = self.data.y_test[:, i_class]
            data_class = Data(self.data.X_train, self.data.X_test,
                              y_train_class, y_test_class)
            # run per-class benchmark
            benchmark_class = self.benchmark_cls(data_class, self.params)
            (tr_time, te_time) = benchmark_class.run()
            # copy the prediction data
            if self.postprocess:
                benchmark_class.y_pred = self.postprocess(benchmark_class.y_pred)
            self.y_pred[:, i_class] = benchmark_class.y_pred
            # aggregate timings
            train_time += tr_time
            test_time += te_time
        return (train_time, test_time)


xgb_common_params = {
    "gamma":            0.1,
    "learning_rate":    0.1,
    "max_depth":        6,
    "max_leaves":       2**6,
    "min_child_weight": 1,
    "num_round":        nTrees,
    "objective":        "binary:logistic",
    "reg_lambda":       1,
    "scale_pos_weight": 2,
    "subsample":        1,
    "benchmark_cls":    XgbBenchmark,
}

lgb_common_params = {
    "learning_rate":    0.1,
    "min_child_weight": 1,
    "min_split_gain":   0.1,
    "num_leaves":       2**6,
    "num_round":        nTrees,
    "objective":        "binary",
    "reg_lambda":       1,
    "scale_pos_weight": 2,
    "subsample":        1,
    "task":             "train",
    "benchmark_cls":    LgbBenchmark,
}

cat_common_params = {
    "depth":            6,
    "iterations":       nTrees,
    "l2_leaf_reg":      0.1,
    "learning_rate":    0.1,
    "loss_function":    "Logloss",
    "benchmark_cls":    CatBenchmark,
    "postprocess":      catLabels,
}


benchmarks = {
    "xgb-cpu":      (True, PlanetBenchmark, metrics,
                     dict(xgb_common_params, tree_method="exact",
                          nthread=nthreads)),
    "xgb-cpu-hist": (True, PlanetBenchmark, metrics,
                     dict(xgb_common_params, nthread=nthreads,
                          grow_policy="lossguide", tree_method="hist")),
    "xgb-gpu":      (True, PlanetBenchmark, metrics,
                     dict(xgb_common_params, tree_method="gpu_exact",
                          objective="gpu:binary:logistic")),
    "xgb-gpu-hist": (True, PlanetBenchmark, metrics,
                     dict(xgb_common_params, tree_method="gpu_hist", max_bins=63,
                          objective="gpu:binary:logistic")),

    "lgbm-cpu":     (True, PlanetBenchmark, metrics,
                     dict(lgb_common_params, nthread=nthreads)),
    "lgbm-gpu":     (True, PlanetBenchmark, metrics,
                     dict(lgb_common_params, device="gpu", max_bin=63)),

    "cat-cpu":      (True, PlanetBenchmark, metrics,
                     dict(cat_common_params, thread_count=nthreads)),
    "cat-gpu":      (True, PlanetBenchmark, metrics,
                     dict(cat_common_params, task_type="GPU")),
}
