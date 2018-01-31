FROM pytorch/pytorch:latest
RUN apt-get update && apt-get install -y --no-install-recommends libicu-dev locales locales-all wget unzip
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
