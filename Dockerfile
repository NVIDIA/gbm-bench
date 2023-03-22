ARG RAPIDS_VER="21.06"
ARG CUDA_VER="11.0"
ARG IMG_TYPE="devel"
ARG LINUX_VER="ubuntu18.04"
ARG PYTHON_VER="3.8"

FROM rapidsai/rapidsai-dev:${RAPIDS_VER}-cuda${CUDA_VER}-${IMG_TYPE}-${LINUX_VER}-py${PYTHON_VER}

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential

RUN . /opt/conda/etc/profile.d/conda.sh \
    && conda activate rapids

# catboost
RUN if [ "`echo $CUDA_VERSION | sed -e 's/[.].*//'`" -lt "11" ]; then git config --global http.sslVerify false && \
    git clone --recursive "https://github.com/catboost/catboost" /opt/catboost && \
    cd /opt/catboost && \
    cd catboost/python-package/catboost && \
    ../../../ya make \
        -r \
        -o ../../.. \
        -DUSE_ARCADIA_PYTHON=no \
        -DUSE_SYSTEM_PYTHON={PYTHON_VER}\
        -DPYTHON_CONFIG=python3-config \
        -DCUDA_ROOT=$(dirname $(dirname $(which nvcc))); \
        fi
ENV if [ "`echo $CUDA_VERSION | sed -e 's/[.].*//'`" -lt "11" ]; then PYTHONPATH=$PYTHONPATH:/opt/catboost/catboost/python-package; fi

# xgboost
RUN git config --global http.sslVerify false && \
    git clone --recursive https://github.com/dmlc/xgboost /opt/xgboost && \
    cd /opt/xgboost && \
    mkdir build && \
    cd build && \
    RMM_ROOT=/opt/conda/envs/rapids cmake .. \
        -DUSE_CUDA=ON \
        -DUSE_NCCL=ON \
        -DPLUGIN_RMM=ON && \
    make -j4 && \
    cd ../python-package && \
    pip uninstall -y xgboost && \
    python setup.py install
