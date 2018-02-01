FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04 
RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         vim \
         ca-certificates \
         libnccl2=2.0.5-2+cuda8.0 \
         libnccl-dev=2.0.5-2+cuda8.0 \
         libjpeg-dev \
         libpng-dev &&\
     rm -rf /var/lib/apt/lists/*
ENV PYTHON_VERSION=3.6
RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \     
     rm ~/miniconda.sh && \
#     /opt/conda/bin/conda install conda-build && \
     /opt/conda/bin/conda clean -ya 
ENV PATH=/opt/conda/bin:${PATH}
RUN conda install -y pytorch torchvision -c pytorch
RUN apt-get update && apt-get install -y --no-install-recommends libicu-dev locales locales-all wget unzip\
    autoconf automake libtool pkg-config libprotobuf9v5 protobuf-compiler libprotobuf-dev
RUN mkdir /sentencepiece
WORKDIR /sentencepiece
RUN git clone https://github.com/google/sentencepiece.git .
RUN ./autogen.sh
RUN ./configure
RUN make && make check && make install && ldconfig -v
ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8
RUN pip install polyglot
RUN mkdir /work
WORKDIR /work
RUN wget https://github.com/facebookresearch/fastText/archive/v0.1.0.zip
RUN unzip v0.1.0.zip && mv fastText-0.1.0 fastText
WORKDIR fastText
RUN make 
WORKDIR ..
RUN git clone https://github.com/facebookresearch/MUSE.git
RUN conda install -y -c pytorch faiss-gpu
COPY Makefile .
