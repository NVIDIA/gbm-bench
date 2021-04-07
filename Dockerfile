ARG CUDA_VERSION
FROM nvidia/cuda:$CUDA_VERSION-devel-ubuntu18.04
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
RUN conda install -c conda-forge -c rapidsai -c nvidia -c defaults \
        bokeh \
        cmake>=3.14 \
        h5py \
        ipython \
        ipywidgets \
        jupyter \
        kaggle \
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
        tqdm \
        cudf=0.18.0 \
        dask-cuda \
        rmm \
        librmm \
        rapids-xgboost \
        cuml=0.18 && \
    conda clean -ya

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
RUN if [ "`echo $CUDA_VERSION | sed -e 's/[.].*//'`" -lt "11" ]; then git config --global http.sslVerify false && \
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
ENV if [ "`echo $CUDA_VERSION | sed -e 's/[.].*//'`" -lt "11" ]; then PYTHONPATH=$PYTHONPATH:/opt/catboost/catboost/python-package; fi



# xgboost
RUN git config --global http.sslVerify false && \
    git clone --recursive https://github.com/dmlc/xgboost /opt/xgboost && \
    cd /opt/xgboost && \
    mkdir build && \
    cd build && \
    RMM_ROOT=/opt/conda cmake .. \
        -DUSE_CUDA=ON \
        -DUSE_NCCL=ON \
        -DPLUGIN_RMM=ON && \
    make -j4 && \
    cd ../python-package && \
    pip uninstall -y xgboost && \
    python setup.py install
