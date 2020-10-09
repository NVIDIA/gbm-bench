ARG CUDA_VERSION
FROM nvidia/cuda:$CUDA_VERSION-devel-ubuntu16.04
SHELL ["/bin/bash", "-c"]
# Install conda (and use python 3.7)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        doxygen \
        git \
        graphviz \
        libcurl4-openssl-dev \
        libboost-all-dev \
        make \
        tar \
        unzip \
        wget \
        zlib1g-dev && \
    rm -rf /var/lib/apt/*

RUN curl -o /opt/miniconda.sh \
	https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x /opt/miniconda.sh && \
    /opt/miniconda.sh -b -p /opt/conda && \
    /opt/conda/bin/conda update -n base conda && \
    rm /opt/miniconda.sh
ENV PATH /opt/conda/bin:$PATH
RUN conda install -c conda-forge \
        bokeh \
        h5py \
        ipython \
        ipywidgets \
        jupyter \
        matplotlib \
        nose \
        numpy \
        pandas \
        Pillow \
        pydot \
        pylint\
        psutil\
        scikit-learn \
        scipy \
        six \
        dask \
        distributed \
        tqdm && \
        conda clean -ya && \
        pip install kaggle tqdm && \
        conda install -c rapidsai -c nvidia -c conda-forge -c defaults cudf=0.15.0 dask-cuda rmm librmm rapids-xgboost cuml=0.15

# cmake
ENV CMAKE_SHORT_VERSION 3.14
ENV CMAKE_LONG_VERSION 3.14.7
RUN wget --no-check-certificate \
        "https://cmake.org/files/v${CMAKE_SHORT_VERSION}/cmake-${CMAKE_LONG_VERSION}.tar.gz" && \
    tar xf cmake-${CMAKE_LONG_VERSION}.tar.gz && \
    cd cmake-${CMAKE_LONG_VERSION} && \
    ./bootstrap --system-curl && \
    make -j && \
    make install && \
    cd .. && \
    rm -rf cmake-${CMAKE_LONG_VERSION}.tar.gz cmake-${CMAKE_LONG_VERSION}

# lightgbm
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        bzip2 \
        ca-certificates \
        curl \
        git \
        libblas-dev \
        libboost-dev \
        libboost-filesystem-dev \
        libboost-system-dev \
        libbz2-dev \
        libc6 \
        libglib2.0-0 \
        liblapack-dev \
        libsm6 \
        libxext6 \
        libxrender1 \
        make \
        tar \
        unzip \
        wget && \
    rm -rf /var/lib/apt/*
RUN mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd
ENV OPENCL_LIBRARIES /usr/local/cuda/lib64
ENV OPENCL_INCLUDE_DIR /usr/local/cuda/include
RUN git config --global http.sslVerify false && \
    git clone --recursive https://github.com/Microsoft/LightGBM /opt/LightGBM && \
    cd /opt/LightGBM && \
    mkdir build && \
    cd build && \
    cmake .. \
        -DUSE_GPU=1 \
        -DOpenCL_LIBRARY=$OPENCL_LIBRARIES/libOpenCL.so \
        -DOpenCL_INCLUDE_DIR=$OPENCL_INCLUDE_DIR && \
    make OPENCL_HEADERS="/usr/local/cuda/targets/x86_64-linux/include" \
        LIBOPENCL="/usr/local/cuda/targets/x86_64-linux/lib" -j4 && \
    cd ../python-package && \
    python setup.py install --precompile

# catboost
RUN if ["$CUDA_VERSION" < "11.0"]; then git config --global http.sslVerify false && \
    git clone --recursive "https://github.com/catboost/catboost" /opt/catboost && \
    cd /opt/catboost && \
    cd catboost/python-package/catboost && \
    ../../../ya make \
        -r \
        -o ../../.. \
        -DUSE_ARCADIA_PYTHON=no \
        -DUSE_SYSTEM_PYTHON=3.7\
        -DPYTHON_CONFIG=python3-config \
        -DCUDA_ROOT=$(dirname $(dirname $(which nvcc))); \
        fi
ENV if ["$CUDA_VERSION" < "11.0"]; then PYTHONPATH=$PYTHONPATH:/opt/catboost/catboost/python-package; fi\



# xgboost
RUN git config --global http.sslVerify false && \
    git clone --recursive https://github.com/dmlc/xgboost /opt/xgboost && \
    cd /opt/xgboost && \
    mkdir build && \
    cd build && \
    RMM_ROOT=/opt/conda cmake .. \
        -DGPU_COMPUTE_VER="60;70;80" \
        -DUSE_CUDA=ON \
        -DUSE_NCCL=ON \
        -DPLUGIN_RMM=ON && \
    make -j4 && \
    cd ../python-package && \
    pip uninstall -y xgboost && \
    python setup.py install
