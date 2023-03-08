# Visidon_Image_Translation_Test


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

The inference file ```inference.py``` will randomly choose, process, and display image one-by-one from the folder defined by
```dataroot```. 

```
python inference.py
```

