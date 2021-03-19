FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

ADD . /color_extraction/

RUN apt-get update
RUN apt-get install -y python3-dev
RUN apt-get install -y python3-pip
RUN pip3 install --upgrade pip

RUN pip3 install Cython
RUN pip3 install -r requirements.txt