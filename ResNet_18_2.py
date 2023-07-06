### 라이브러리

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, MaxPooling2D, AveragePooling2D, Flatten, Dense
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import cv2
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras.backend as K

#%% GPU 할당 

# GPU를 사용할 수 있는지 확인
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) == 0:
    print("GPU를 찾을 수 없습니다. CPU를 사용합니다.")
else:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU를 사용합니다.")


#%% block설정

def resnet_block(inputs,kernel,filters,strides=1,activation='relu'):
    x=Conv2D(filters,kernel_size=(kernel,kernel),strides=strides, padding='same')(inputs)
    x=BatchNormalization()(x)
    x=Activation(activation)(x)
    x=Conv2D(filters,kernel_size=(kernel,kernel),padding='same')(x)
    x=BatchNormalization()(x)
    if strides != 1 or inputs.shape[3] != filters:
       inputs = Conv2D(filters, kernel_size=(1, 1), strides=strides, padding='same')(inputs)
       inputs = BatchNormalization()(inputs)
    x=Add()([inputs,x])
    x=Activation(activation)(x)
    return x


#%% 모델 설계

def resnet_18(input_shape,num_classes):
    inputs=Input(shape=input_shape)
    x=Conv2D(64,kernel_size=(7,7),strides=(2,2),padding='same')(inputs)
    x=BatchNormalization()(x)
    x = Activation('relu')(x)
    x=MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)
    x=resnet_block(x,3,64)
    x=resnet_block(x,3,64)
    x=resnet_block(x,3,128,strides=2)
    x=resnet_block(x,3,128)
    
    
    x=resnet_block(x,3,256,strides=2)
    x=resnet_block(x,3,256)
    x=resnet_block(x,3,512,strides=2)
    x=resnet_block(x,3,512)
    
    
    x=AveragePooling2D(pool_size=(7, 7), strides=(1, 1))(x)
    x=Flatten()(x)
    x=Dense(num_classes,activation='softmax')(x)
    
    model=Model(inputs=inputs,outputs=x)
    return model



#%% 이미지의 크기와 클래스 갯수 설정

input_shape=(224,224,3) #이미지의 크기
num_classes=5 #클래스의 숫자


#%% 모델 요약

## resnet

model=resnet_18(input_shape,num_classes)


#%% 모델 요약

model.summary()


#%% 이미지의 경로 (수정은 여기!!)

image_path="C:/Users/VSA/Desktop/L_TL2/cat/eye/normal"

categories=["a","b","c","d","e"]
X=[]
Y=[]


#%% 이미지 로드



for idx, cate in enumerate(categories):
    label=[0 for i in range(num_classes)]
    label[idx]=1
    image_dir=image_path+ '/' + cate + '/Y/'
    
    for top,dir,f in os.walk(image_dir):
        i=0
        for filename in f:
            if filename[-4:]!='json':
                i+=1
                img=cv2.imread(image_dir+filename,1)
                img=cv2.resize(img,(200,200))
                img = cv2.copyMakeBorder(img, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=0)
                X.append(img)
                Y.append(label)
                if i==800:
                    break
X=np.array(X)
Y=np.array(Y)


#%% train test셋 정의

x_train,x_test,y_train,y_test= train_test_split(X,Y)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


#%% 모델 적용 및 실행

model.compile(optimizer=SGD(learning_rate=1e-03),loss='categorical_crossentropy',metrics=['accuracy'])
with tf.device('/GPU:0'):
    history=model.fit(x_train,y_train,epochs=7,batch_size=4)


#%% 모델 평가

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


#%% 정리
K.clear_session()
