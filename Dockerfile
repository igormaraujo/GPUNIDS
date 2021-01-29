ARG repository
FROM nvidia/cuda:8.0-devel-ubuntu16.04
LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"

RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-core-$CUDA_PKG_VERSION \
        cuda-misc-headers-$CUDA_PKG_VERSION \
        cuda-command-line-tools-$CUDA_PKG_VERSION \
        cuda-nvrtc-dev-$CUDA_PKG_VERSION \
        cuda-nvml-dev-$CUDA_PKG_VERSION \
        cuda-nvgraph-dev-$CUDA_PKG_VERSION \
        cuda-cusolver-dev-$CUDA_PKG_VERSION \
        cuda-cublas-dev-8-0=8.0.61.2-1 \
        cuda-cufft-dev-$CUDA_PKG_VERSION \
        cuda-curand-dev-$CUDA_PKG_VERSION \
        cuda-cusparse-dev-$CUDA_PKG_VERSION \
        cuda-npp-dev-$CUDA_PKG_VERSION \
        cuda-cudart-dev-$CUDA_PKG_VERSION \
        cuda-driver-dev-$CUDA_PKG_VERSION && \
    rm -rf /var/lib/apt/lists/*

ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs

#Install htop, vim and pre-requirements
RUN apt-get update && apt-get install -y htop vim git wget flex bison make cmake

#Install LIBPCAP
RUN wget http://www.tcpdump.org/release/libpcap-1.9.0.tar.gz \
&& tar -zxvf libpcap-1.9.0.tar.gz \
&& cd libpcap-1.9.0 \
&& ./configure \
&& make \
&& make install

#Install iPerf3
RUN wget https://iperf.fr/download/ubuntu/iperf3_3.1.3-1_amd64.deb \
&& wget https://iperf.fr/download/ubuntu/libiperf0_3.1.3-1_amd64.deb \
&& dpkg -i libiperf0_3.1.3-1_amd64.deb iperf3_3.1.3-1_amd64.deb \
&& rm libiperf0_3.1.3-1_amd64.deb iperf3_3.1.3-1_amd64.deb

#Vim Configurations
RUN git clone http://www.lincpo.com.br:2341/igoraraujo/vim_cpp.git \
&& cp vim_cpp/vimrc ~/.vimrc \ 
&& git clone https://github.com/VundleVim/Vundle.vim.git ~/.vim/bundle/Vundle.vim \
&& vim +PluginInstall +qall

RUN mkdir /cudaworkspace
