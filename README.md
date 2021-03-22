# Source code to Paper 9041

## Requirements
* RAM >= 16 GB
* Free disk space >= 20 GB
* Ubuntu >= 18.04

## Deploy instructions
#### Native deploy
To install dependencies run follow shell commands:
```shell script
sudo apt-get update
sudo apt-get install -y python3-dev
sudo apt-get install -y python3-pip
pip3 install --upgrade pip
sudo apt install -y python3-opencv

pip3 install Cython
pip3 install -r requirements.txt
pip3 install jupyter
```

Next, run Jupyter Notebook by following shell command:
```shell script
jupyter notebook --port=9005
```

#### Docker deploy (recommended)
To build docker image run follow shell command:
```shell script
sudo docker build -t color_extraction_container .
```

To run container run follow shell command:
```shell script
sudo docker run -p 9005:9005 --name ColorExtractionPaperID9041 -d color_extraction_container
```

To destroy docker container run following shell command:
```shell script
sudo docker rm -f ColorExtractionPaperID9041
```

## Usage instructions
#### Inference
First of all, need open the following page in your browser:
`localhost:9005/`

To check example inference code open notebook by path:
```notebooks/inference_example.ipynb```