# Kaggle入门(二)——Dogs vs. Cats

[TOC]

## 0 前言

比赛网址：

https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition

参考解法：

https://www.kaggle.com/jeffd23/catdognet-keras-convnet-starter

https://www.kaggle.com/sentdex/full-classification-example-with-convnet



## 1 简介

- 卷机网络模型——对VGG16修改
- 基于Keras
- test loss: 0.23535
- Titan V，73epoch，训练时间1h+

导包：

```python
import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm      # a nice pretty percentage bar for tasks. Thanks to viewer Daniel Bühler for this suggestion
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau
from keras.utils import np_utils
from keras import backend

%matplotlib inline
```

## 2 数据准备

### 2.1 导入数据

```python
TRAIN_DIR = './data/train'
TEST_DIR = './data/test'
IMG_SIZE = 128

MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR, '2conv-basic')
```
- 将标签变成one-hot编码（直接用0,1做标签，会出问题，在测试集上表现奇差，换成One-hot之后解决，未研究清楚为什么）

```python
# one-hot 编码
def label_img(img):
    word_label = img.split('.')[-3]
    if word_label == 'cat': 
        return [1,0]
    elif word_label == 'dog': 
        return [0,1]
```

- 导入训练和测试数据
  - 将图片转为灰度图
  - 图片尺寸改为IMG_SIZE * IMG_SIZE
  - 将处理后的图片保存为`.npy`格式，方便下次读取
  - 使用tqdm库，可以将处理过程用进度条表示出来 awesome  :smile:

```python
# 处理训练数据
def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img), label])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

# 处理测试数据
def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)
        img_num = img.split('.')[0]
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img), img_num])
        
    #shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data
```

```python
train_data = create_train_data()
# If you have already created the dataset:
#train_data = np.load('train_data.npy')
```

### 2.2 分割验证集
```python
train, val = train_test_split(train_data, test_size = 0.25)

X_train = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y_train = np.array([i[1] for i in train])

X_val = np.array([i[0] for i in val]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y_val = np.array([i[1] for i in val])
```

## 3 Convolutional Neural Network

### 3.1 定义网络模型

**网络结构：**

**基于VGG16**

conv2D(64, (3,3)) -> conv2D(64, (3,3)) -> maxpool(2,2)   
-> conv2D(128, (3,3)) -> conv2D(128, (3,3)) -> maxpool(2,2)  
-> conv2D(256, (3,3)) -> conv2D(256, (3,3)) -> conv2D(256, (3,3)) -> maxpool(2,2)  
-> conv2D(512, (3,3)) -> conv2D(512, (3,3)) -> conv2D(512, (3,3)) -> maxpool(2,2)  
-> conv2D(512, (3,3)) -> conv2D(512, (3,3)) -> conv2D(512, (3,3)) -> maxpool(2,2)   
-> flatten() -> full connect(256) -> dropout(0.5) -> full connect(2)

```python
backend.set_image_dim_ordering('tf')  # th通道在前，tf通道在后

optimizer = RMSprop(lr=1e-4)
objective = 'categorical_crossentropy'
IMG_SIZE = 128

def catdog():
    
    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 1), activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(2, activation='softmax'))

    model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])
    return model

model = catdog()
```

- 模型参数

```python
model.summary()
```

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_24 (Conv2D)           (None, 128, 128, 64)      640       
_________________________________________________________________
conv2d_25 (Conv2D)           (None, 128, 128, 64)      36928     
_________________________________________________________________
max_pooling2d_10 (MaxPooling (None, 64, 64, 64)        0         
_________________________________________________________________
conv2d_26 (Conv2D)           (None, 64, 64, 128)       73856     
_________________________________________________________________
conv2d_27 (Conv2D)           (None, 64, 64, 128)       147584    
_________________________________________________________________
max_pooling2d_11 (MaxPooling (None, 32, 32, 128)       0         
_________________________________________________________________
conv2d_28 (Conv2D)           (None, 32, 32, 256)       295168    
_________________________________________________________________
conv2d_29 (Conv2D)           (None, 32, 32, 256)       590080    
_________________________________________________________________
conv2d_30 (Conv2D)           (None, 32, 32, 256)       590080    
_________________________________________________________________
max_pooling2d_12 (MaxPooling (None, 16, 16, 256)       0         
_________________________________________________________________
conv2d_31 (Conv2D)           (None, 16, 16, 512)       1180160   
_________________________________________________________________
conv2d_32 (Conv2D)           (None, 16, 16, 512)       2359808   
_________________________________________________________________
conv2d_33 (Conv2D)           (None, 16, 16, 512)       2359808   
_________________________________________________________________
max_pooling2d_13 (MaxPooling (None, 8, 8, 512)         0         
_________________________________________________________________
conv2d_34 (Conv2D)           (None, 8, 8, 512)         2359808   
_________________________________________________________________
conv2d_35 (Conv2D)           (None, 8, 8, 512)         2359808   
_________________________________________________________________
conv2d_36 (Conv2D)           (None, 8, 8, 512)         2359808   
_________________________________________________________________
max_pooling2d_14 (MaxPooling (None, 4, 4, 512)         0         
_________________________________________________________________
flatten_3 (Flatten)          (None, 8192)              0         
_________________________________________________________________
dense_5 (Dense)              (None, 256)               2097408   
_________________________________________________________________
dropout_3 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_6 (Dense)              (None, 2)                 514       
=================================================================
Total params: 16,811,458
Trainable params: 16,811,458
Non-trainable params: 0
_________________________________________________________________
```

### 3.2 数据增强

随机：旋转40；错切变换0.3；缩放0.2；左右上下平移0.2；水平垂直翻转

```python
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=40,  # randomly rotate images in the range (degrees, 0 to 180)
        shear_range=0.3,    #错切变换，效果就是让所有点的x坐标(或者y坐标)保持不变，而对应的y坐标(或者x坐标)则按比例发生平移
        zoom_range = 0.2, # Randomly zoom image 
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)  # randomly flip images


#datagen = ImageDataGenerator()
datagen.fit(X_train)
```

### 3.3 训练

- learning_rate_reduction
  - 当val loss不再减小时，缩小LR的值
- early_stopping
  - LR减小到阈值后，且val loss不再减小时，停止训练
- ModelCheckpoint
  - 保存模型。每个epoch训练之后，若val loss比之前小，保存当前权重

```python
epochs = 100
batch_size = 64

learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                            patience=4, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=5e-8)

early_stopping = EarlyStopping(monitor='val_loss', patience=6, min_delta=0.0002, verbose=1, mode='auto')     
filepath="./weights/weights.best.hdf5"
if not os.path.exists('./weights'):
        os.mkdir('./weights')
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
       
        
def run_catdog():
    
    model_his = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,Y_val), shuffle=True, 
                              verbose = 1, steps_per_epoch=X_train.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction, early_stopping, checkpoint])
    '''
    model_his = model.fit(X_train, np.array(Y_train), batch_size=batch_size, epochs=epochs,
              validation_split=0.2, verbose=1, shuffle=True, callbacks=[learning_rate_reduction, early_stopping, checkpoint])

    '''
    return model_his

history = run_catdog()
```

## 4 评估模型

### 4.1 训练和交叉验证曲线

```python
# Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)
```

![](https://pic.superbed.cn/item/5c9041eb3a213b0417ba0fec)

- accuracy接近92%，loss 0.2左右
- 模型没有过拟合
- 进一步提高精度、减小loss，可通过增加网络层数或使用迁移学习

## 5 生成结果

```python
# if you need to create the data:
test_data = process_test_data()
# if you already have some saved:
#test_data = np.load('test_data.npy')
```

```python
with open('submission_file.csv','w') as f:
    f.write('id,label\n')
            
with open('submission_file.csv','a') as f:
    for data in tqdm(test_data):
        img_num = data[1]
        img_data = data[0]
        orig = img_data
        data = img_data.reshape(-1, IMG_SIZE,IMG_SIZE,1)
        model_out = model.predict([data])[0]
        f.write('{},{}\n'.format(img_num,model_out[1]))
```

## 6 踩过的坑

【待完成】

## 