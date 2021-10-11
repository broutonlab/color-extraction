FROM ubuntu:18.04

ADD . /color_extraction/

RUN apt-get update
RUN apt-get install -y python3-dev
RUN apt-get install -y python3-pip
RUN pip3 install --upgrade pip

RUN apt install -y build-essential cmake unzip pkg-config
RUN apt install -y libjpeg-dev libpng-dev libtiff-dev
RUN apt install -y software-properties-common
RUN add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main"
RUN apt update
RUN apt install -y libjasper1 libjasper-dev
RUN apt install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
RUN apt install -y libxvidcore-dev libx264-dev
RUN apt install -y libgl1-mesa-glx

RUN pip3 install numpy

RUN pip3 install Cython
WORKDIR /color_extraction/
RUN pip3 install -r requirements.txt
RUN pip3 install jupyter

WORKDIR /color_extraction/
CMD ["jupyter", "notebook", "--port=9005", "--no-browser", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''"]