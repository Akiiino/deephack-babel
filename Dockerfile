FROM pytorch/pytorch:latest
RUN sudo apt-get install autoconf automake libtool pkg-config libprotobuf9v5 protobuf-compiler libprotobuf-dev
RUN mkdir /sentencepiece
WORKDIR /sentencepiece
RUN git clone https://github.com/google/sentencepiece.git .
RUN ./autogen.sh
RUN ./configure
RUN make && make check && sudo make install && sudo ldconfig -v
