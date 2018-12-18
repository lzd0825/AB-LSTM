# AB-LSTM: Attention-Based Bidirectional LSTM Model for Scene Text Detection
By Zhandong Liu, Wengang Zhou and Houqiang Li.  

# 1. Introduction
This project contains the following source files: model training and testing, text center block label and word stroke region label generation, label augmentation, and sample models that have been trained.
# 2. Installation
* Clone the repo  
```
git clone https://github.com/lzd0825/AB-LSTM.git
cd ./AB-LSTM
```
* Requirements for `Caffe` and `pycaffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))

  **Note:** Caffe *must* be built with support for Python layers!

  ```make
  # In your Makefile.config, make sure to have this line uncommented
  WITH_PYTHON_LAYER := 1
  # Unrelatedly, it's also recommended that you use CUDNN
  USE_CUDNN := 1
  ```
  
* Then you can do as follow:
```
cd ./AB-LSTM/Train_Test_ABLSTM/caffe/}  
make –j  
make pycaffe 
```
# 3. Testing
## 3.1 Generate WSR/TCB  score map
* Download the [TD_Total_Text_WSR_iter_175000.caffemodel](https://pan.baidu.com/s/14fg4tR0dim_OiMC7siBtnw), trained on Total-text training dataset.
* Download the [TD_ICDAR2013_TCB_iter_50000.caffemodel](https://pan.baidu.com/s/1ZbFKsXmGbSfzWSnZ9WBx4w), finute trained on ICDAR2013 training dataset.  
* Then you can do as follow:
```
cd ../snapshot  
```
* Put both trained caffemodels to the fold of ${/AB-LSTM/Train_Test_ABLSTM/snapshot}.  

* Suppose you have downloaded the MSRA-TD500 dataset, execute the following commands to test the model on MSRA-TD500.  Then you can do as follow:
```
cd ../Demo  
python Demo_forword_TCB.py  
python Demo_forword_WSR.py  
```
## 3.2 There are some samples:  

![image](https://github.com/lzd0825/AB-LSTM/blob/master/Demo_Text_detection/Data/Forword/some_forwords.jpg)


## 3.3 Threshold WSR/TCB maps:
You can do as follow:
```
cd ${AB-LSTM/Demo_Text_detection}  
python fuse_thred.py  
```
## 3.4 Generate detection results  
You can do as follow:
```
python Demo_region_word.py
```
## 3.5 There are some samples:

![image](https://github.com/lzd0825/AB-LSTM/blob/master/Demo_Text_detection/Data/save_detection/some_results.jpg)

# 4. Training
Download the pretrained model [vgg16convs.caffemodel](https://pan.baidu.com/s/1IEt48THcdmncH2zoeokypA), and put it to 
${AB-LSTM/Train_Test_ABLSTM/model/}

## 4.1 Generate your TCB label and WSR label  
Scripts for generating ground truth have been provided in the ${AB-LSTM/Label_generate}. You can use our code to generate you own training labels on different public datasets (e.g. ICDAR2013, MSRA-TD500, CTW1500, and Total-text, etc.).

## 4.2 Data Augmentation
We use “ImageDataGenerator” in “keras.preproces-sing.image” to achieve data augmentation.
cd ${AB-LSTM/Data_aug}

You must modify the parameters image_save_prefix and mask_save_prefix in the trainGenerator function. Note that you must use an absolute path, such as: image_save_prefix = "/data1/XXX/aug_dataset/Aug_example/train_aug/aug",mask_save_prefix = "/data1/XXX/aug_dataset/Aug_example /train_gt_aug/aug".

## There are some samples on data augmentation:
![image](https://github.com/lzd0825/AB-LSTM/blob/master/Data_Aug/Aug_exmple/data_aug.jpg)

<img style="margin: auto;" src="https://github.com/lzd0825/AB-LSTM/blob/master/Data_Aug/Aug_exmple/data_aug.jpg" width=80%>

## 4.3 Train your own model
Modify ${AB-LSTM/Train_Test_ABLSTM/TD_ICDAR2013_TCB.py, and TD_Total_Text_WSR.py} to configure your dataset name and dataset path like:  
......  
data_params['root'] = "./AB-LSTM/Train_Test_ABLSTM/datasets/Total_Text_WSR/"
data_params['source'] = "Total_Text_WSR.lst"  
......

## 4.4 Start training

You can do as follow:
```
cd ${AB-LSTM/Train_Test_ABLSTM/}  
sh ./train_ICDAR2013_TCB.sh
sh ./train_Total_Text_WSR.sh

```
## Citation
Use this bibtex to cite this repository:
```
@misc{liu_AB-LSTM_2018,
  title={AB-LSTM: Attention-Based Bidirectional LSTM Model for Scene Text Detection},
  author={Zhandong Liu, Wengang Zhou, Houqiang Li},
  year={2018},
  publisher={Github},
  journal={GitHub repository},
  howpublished={\url{https://github.com/lzd0825/AB-LSTM/}},
}
```
# Acknowlegement
