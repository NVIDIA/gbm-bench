ARG RAPIDS_VER="21.06"
ARG CUDA_VER="11.0"
ARG IMG_TYPE="base"
ARG LINUX_VER="ubuntu18.04"
ARG PYTHON_VER="3.8"
FROM rapidsai/rapidsai-core:${RAPIDS_VER}-cuda${CUDA_VER}-${IMG_TYPE}-${LINUX_VER}-py${PYTHON_VER}

ENV PYTHONUNBUFFERED=True

RUN apt update -y \
    && apt install -y --no-install-recommends build-essential \
    && apt install -y cmake \
    && apt autoremove -y \
    && apt clean -y \
    && rm -rf /var/lib/apt/lists/*

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
