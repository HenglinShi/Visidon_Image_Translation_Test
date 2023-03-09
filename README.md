# Visidon_Image_Translation_Test

## Abstract

This projects performs end-to-end image transformation using deep neural networks.
The architecture of the model is a 2D autoencoder, where the encoder and decoder are implemented using the ResNet18-like
architecture.
The network model is implemented based on existing resources from the Internet.
Currently, there are many public repositories that implement 2D encoder-decoder architectures, I adopted the one
from: ```https://github.com/julianstastny/VAE-ResNet18-PyTorch/blob/master/model.py.```
The loss function used to supervise the modeling training is the ```squared L2 norm```.

The performance of the current implementation is not ideal, because the transformed images are a bit blurry, but the
targeted image
should be more sharp. I am planning to use ```L1 norm``` as the loss function in the next step.

Simplying to see the result please refer to [\[inference.ipynb\]](inference.py)

## Getting Started

### Environment

- python 3.7
- pytorch 1.10.0
- opencv-python 4.6.0.66
- scikit-image 0.19.3
- matplotlib 3.5.3
- tqdm 4.64.1

### Setup

```shell script
pip3 install -r requirements.txt
```

### Data preparation

#### Training

Downloading the dataset from https://www.visidon.fi/wp-content/uploads/2022/11/vd_test2.pdf, extracting the data,
and placing the folder in the root directory of the project. By default, name of the data folder will be 'VD_dataset2',
but please change the variable ```dataroot``` in the ```train.py``` accordingly.

```
python train.py
```

#### Inference

The inference file ```inference.py``` will randomly choose, process, and display image one-by-one from the folder
defined by
```dataroot```.

```
python inference.py
```

Result visualization is interactive, after clicking on the image, a new image will be processed and displayed.
