FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# install python
ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.10
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false

#for some regions, it's useful
ENV LANG C.UTF-8
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub


RUN apt-get update && \
  apt-get -qq -y install \
  software-properties-common \
  build-essential \
  curl \
  ffmpeg \
  git \
  htop \
  swig \
  vim \
  nano \
  rsync \
  tmux \
  wget \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y -qq python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-distutils

# Set python aliases and getting the latest pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python get-pip.py

EXPOSE 9999

# default workdir
WORKDIR /home/workdir

COPY ./requirements.txt .

RUN pip install -r requirements.txt --ignore-installed
RUN pip install tensorflow[and-cuda]==2.14
RUN pip install --upgrade jax==0.4.25 jaxlib==0.4.25+cuda11.cudnn86 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

RUN git config --global --add safe.directory /home/workdir
CMD ["/bin/bash"]

