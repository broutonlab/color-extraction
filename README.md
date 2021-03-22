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
## Repository structure
```bash 
.
├── Dockerfile
├── evaluate_on_data.py
├── evaluation_data
│   ├── images
│   └── masks
├── examples
├── notebooks
│   ├── inference_example.ipynb
│   └── pipline_steps.ipynb
├── README.md
├── requirements.txt
├── third_party
└── utils
    ├── array_utils.py
    ├── color_embedding.py
    ├── evaluation_utils.py
    ├── topological_colors_extraction.py
    └── visualization.py
```
1. `evaluation_data` contains 28 original images for evaluation in `evaluation_data/images` and 28 labeled images for measuring mIOU in `evaluation_data/masks`
2. `examples` contains images on which you can check the operation of the algorithm 
3. `third_party` includes [pallete embedding model](https://github.com/googleartsculture/art-palette) and [umato](https://github.com/hyungkwonko/umato)  
4. `utils` - implementation of our method 

## Usage instructions
#### Inference
First of all, need open the following page in your browser:
`localhost:9005/`

To check example inference code open notebook by path:
```notebooks/inference_example.ipynb```

To check the algorithm step by step open: `notebooks/pipline_steps.ipynb`, this notebook also contains the running time of each stage 
#### Evaluation 
To evaluate the algorithm on all 28 images run: `evaluate_on_data.py --low_memory=False`  
Or open ```notebooks/inference_example.ipynb``` and run   
```!python evaluate_on_data.py --low_memory=False```   
If system has less then 16 RAM, set `--low_memory=True`, default is `False`