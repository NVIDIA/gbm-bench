# Introduction
This repo tries to benchmark boosting frameworks against some of the popular
ML datasets.

# Setting up this repo
```bash
  # make sure that you have CUDA SDK (or atleast nvidia driver) installed
  # install docker and nvidia-docker, first

  # dockerfiles github project is needed to build the docker image
  $ git clone https://github.com/teju85/dockerfiles
  $ cd dockerfiles/ubuntu1604
  $ make xgb-lgb

  # this project
  $ cd ../..
  $ git clone https://github.com/teju85/gbm-perf   ## TODO!

  $ cd ..
  # Download the datasets as described in the next section.
```

# Datasets
This section contains info on how to download the datasets for this exercise.
Note that to download some datasets, one may have to sign-up in the respective
websites and probably also have to agree to their T&C's! Also note that to
maintain brevity, the dataset description is omitted out of this document.
Interested readers would want to visit respective links to know more about it.

At first, create a root folder which contains all the datasets that will be
downloaded here. Let's calls it 'gbm-datasets'. From here onwards, unless
otherwise mentioned, all folders and paths will be wrt this root folder only.
```bash
$ mkdir boosting-datasets
```

## Airline
Airline dataset and more info on it can be found here: http://kt.ijs.si/elena_ikonomovska/data.html
```bash
$ mkdir airline
$ cd airline
$ wget http://kt.ijs.si/elena_ikonomovska/datasets/airline/airline_14col.data.bz2
```

## Allstate
This is the dataset in the Kaggle-AllState claim prediction challenge.
Link for the dataset is here: https://www.kaggle.com/c/ClaimPredictionChallenge/data
Download the 'dictionary.html' and 'train_set.zip' files from this page inside a
directory named 'allstate'.

## BCI
Brain Computer Interaction dataset.
This is something I'm not sure where the MS researchers got from!
I tried a few trivial google-search attempts with no success.

## Bosch
Bosch dataset as given out on Kaggle competition here: https://www.kaggle.com/c/bosch-production-line-performance/data
Download the files 'train_numeric.csv.zip', 'train_date.csv.zip' and 'train_categorial.csv.zip'
into a folder named 'bosch'.

## Criteo
Criteo display ad challenge as found here: https://www.kaggle.com/c/criteo-display-ad-challenge
```bash
$ mkdir criteo
$ cd criteo
$ wget https://s3-eu-west-1.amazonaws.com/criteo-labs/dac.tar.gz
```

## Football
Football dataset and its info can be found on Kaggle here: https://www.kaggle.com/hugomathien/soccer
Download the 'soccer.zip' from this page into a directory named 'football'.
Code used to process this dataset is shamelessly borrowed from here: https://www.kaggle.com/airback/match-outcome-prediction-in-football

## Fraud Detection
Credit Card Fraud Detection Kaggle competition as found here: https://www.kaggle.com/dalpozz/creditcardfraud
Download the creditcardfraud.zip from this page into a folder named 'fraud'.

## Higgs dataset
This is the dataset as found in the UCI repo here: https://archive.ics.uci.edu/ml/datasets/HIGGS
```bash
$ mkdir higgs
$ cd higgs
$ wget https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz
```

## IOT
Sensor stream data from Berkeley as found here: http://www.cse.fau.edu/~xqzhu/stream.html
```bash
$ mkdir iot
$ cd iot
$ wget http://www.cse.fau.edu/~xqzhu/Stream/sensor.arff
```

## Planet: Understanding the Amazon from Space
Planet-Kaggle competition, as found here: https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/data
Download the 'train_v2.csv.zip' from this page into a directory named 'planet'.

# Running benchmarks
```bash
  $ ./dockerfiles/scripts/launch -user xgb-lgb:latest /bin/bash
  user@container$ cd /work/gbm-perf
  user@container$ ./runme.py -root ../gbm-datasets -dataset football
  user@container$ exit
  $ cat ./gbm-perf/football.json
```
