### 라이브러리

from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, MaxPooling2D, AveragePooling2D, Flatten, Dense
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import cv2
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


### block설정

def resnet_block(inputs,kernel,filters,strides=1,activation='relu'):
    x=Conv2D(filters,kernel_size=(kernel,kernel),strides=strides, padding='same')(inputs)
    x=BatchNormalization()(x)
    x=Activation(activation)(x)
    x=Conv2D(filters,kernel_size=(kernel,kernel),padding='same')(x)
    x=BatchNormalization()(x)
    if strides != 1 or inputs.shape[3] != filters:
       inputs = Conv2D(filters, kernel_size=(1, 1), strides=strides, padding='same')(inputs)
    x=Add()([inputs,x])
    x=Activation(activation)(x)
    return x


### 모델 설계

def resnet_18(input_shape,num_classes):
    inputs=Input(shape=input_shape)
    x=Conv2D(64,kernel_size=(7,7),strides=(2,2),padding='same')(inputs)
    x=MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)
    x=resnet_block(x,3,64)
    x=resnet_block(x,3,64)
    x=resnet_block(x,3,128)
    x=resnet_block(x,3,128)
    
    x=resnet_block(x,7,64)
    x=resnet_block(x,7,64)
    x=resnet_block(x,7,64)
    x=resnet_block(x,7,64)
    
    x=AveragePooling2D(pool_size=(7,7))(x)
    x=Flatten()(x)
    x=Dense(num_classes,activation='softmax')(x)
    
    model=Model(inputs=inputs,outputs=x)
    return model


### 이미지의 크기와 클래스 갯수 설정

input_shape=(200,200,3) #이미지의 크기
num_classes=5 #클래스의 숫자


### 모델 요약

model=resnet_18(input_shape,num_classes)
model.summary()


### 이미지의 경로 (수정은 여기!!)

image_path="C:/Users/VSA/Desktop/L_TL2/cat/eye/normal"

categories=["a","b","c","d","e"]
X=[]
Y=[]


### 이미지 로드

label=[0 for i in range(num_classes)]

for idx, cate in enumerate(categories):
    label[idx]=1
    image_dir=image_path+ '/' + cate + '/Y/' #파일 수정해야하는 부분
    
    for top,dir,f in os.walk(image_dir):
        for filename in f:
            if filename[-4:]!='json':
                img=cv2.imread(image_dir+filename,1)
                img=cv2.resize(img,(200,200))
                X.append(img)
                Y.append(label)
X=np.array(X)
Y=np.array(Y)


### train test셋 정의

x_train,x_test,y_train,y_test= train_test_split(X,Y)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


### 모델 적용 및 실행

model.compile(optimizer=Adam(learning_rate=0.01),loss='categorical_crossentropy',metrics=['accuracy'])
history=model.fit(x_train,y_train,epochs=100,batch_size=4)


### 모델 평가

test_loss,test_acc=model.evaluate(x_test,y_test)
print("test accuracy:",test_acc)

plt.plot(history.history['accuracy'])
plt.title('Train Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()

plt.plot(history.history['loss'])
plt.title('TrainLoss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

