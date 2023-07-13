### 라이브러리

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,GlobalAveragePooling2D, Conv2D, BatchNormalization, Activation, Add, MaxPooling2D, AveragePooling2D, Flatten, Dense, ZeroPadding2D
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

from keras.preprocessing.image import ImageDataGenerator

#%% GPU 할당

# GPU를 사용할 수 있는지 확인
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) == 0:
    print("GPU를 찾을 수 없습니다. CPU를 사용합니다.")
else:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU를 사용합니다.")



from google.colab import drive    # google drive mount
drive.mount('/content/drive')

def conv1_layer(x):
  x=ZeroPadding2D(padding=(3, 3))(x)
  x=Conv2D(64,(7,7),strides=(2,2))(x)
  x=BatchNormalization()(x)
  x=Activation('relu')(x)
  x=ZeroPadding2D(padding=(1,1))(x)
  return x

def conv2_layer(x):
    x=MaxPooling2D((3, 3), 2)(x)

    shortcut=x

    for i in range(3):
        if (i == 0):
            x=Conv2D(64, (1, 1), strides=(1, 1), padding='valid')(x)
            x=BatchNormalization()(x)
            x=Activation('relu')(x)

            x=Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
            x=BatchNormalization()(x)
            x=Activation('relu')(x)

            x=Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)
            shortcut=Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(shortcut)
            x=BatchNormalization()(x)
            shortcut=BatchNormalization()(shortcut)

            x=Add()([x, shortcut])
            x=Activation('relu')(x)

            shortcut=x

        else:
            x=Conv2D(64, (1, 1), strides=(1, 1), padding='valid')(x)
            x=BatchNormalization()(x)
            x=Activation('relu')(x)

            x=Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
            x=BatchNormalization()(x)
            x=Activation('relu')(x)

            x=Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)
            x=BatchNormalization()(x)

            x=Add()([x, shortcut])
            x=Activation('relu')(x)

            shortcut=x

    return x

def conv3_layer(x):
    shortcut = x

    for i in range(4):
        if(i == 0):
            x = Conv2D(128, (1, 1), strides=(2, 2), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(x)
            shortcut = Conv2D(512, (1, 1), strides=(2, 2), padding='valid')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

        else:
            x = Conv2D(128, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

    return x

def conv4_layer(x):
    shortcut=x

    for i in range(6):
        if(i == 0):
            x=Conv2D(256, (1, 1), strides=(2, 2), padding='valid')(x)
            x=BatchNormalization()(x)
            x=Activation('relu')(x)

            x=Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
            x=BatchNormalization()(x)
            x=Activation('relu')(x)

            x=Conv2D(1024, (1, 1), strides=(1, 1), padding='valid')(x)
            shortcut=Conv2D(1024, (1, 1), strides=(2, 2), padding='valid')(shortcut)
            x=BatchNormalization()(x)
            shortcut=BatchNormalization()(shortcut)

            x=Add()([x, shortcut])
            x=Activation('relu')(x)

            shortcut=x

        else:
            x=Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)
            x=BatchNormalization()(x)
            x=Activation('relu')(x)

            x=Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
            x=BatchNormalization()(x)
            x=Activation('relu')(x)

            x=Conv2D(1024, (1, 1), strides=(1, 1), padding='valid')(x)
            x=BatchNormalization()(x)

            x=Add()([x, shortcut])
            x=Activation('relu')(x)

            shortcut=x

    return x

def conv5_layer(x):
    shortcut = x

    for i in range(3):
        if(i == 0):
            x=Conv2D(512, (1, 1), strides=(2, 2), padding='valid')(x)
            x=BatchNormalization()(x)
            x=Activation('relu')(x)

            x=Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
            x=BatchNormalization()(x)
            x=Activation('relu')(x)

            x=Conv2D(2048, (1, 1), strides=(1, 1), padding='valid')(x)
            shortcut=Conv2D(2048, (1, 1), strides=(2, 2), padding='valid')(shortcut)
            x=BatchNormalization()(x)
            shortcut=BatchNormalization()(shortcut)

            x=Add()([x, shortcut])
            x=Activation('relu')(x)

            shortcut=x

        else:
            x=Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(x)
            x=BatchNormalization()(x)
            x=Activation('relu')(x)

            x=Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
            x=BatchNormalization()(x)
            x=Activation('relu')(x)

            x=Conv2D(2048, (1, 1), strides=(1, 1), padding='valid')(x)
            x=BatchNormalization()(x)

            x=Add()([x, shortcut])
            x=Activation('relu')(x)

            shortcut=x

    return x

def resnet50(input_shape,num_classes):
  inputs=Input(shape=input_shape)
  x = conv1_layer(inputs)
  x = conv2_layer(x)
  x = conv3_layer(x)
  x = conv4_layer(x)
  x = conv5_layer(x)
  x = GlobalAveragePooling2D()(x)
  x = Dense(num_classes, activation='softmax')(x)
  model=Model(inputs=inputs,outputs=x)
  return model

# Parameter

input_shape=(224,224,3) #이미지의 크기
num_classes=6 #클래스의 숫자

## resnet
model=resnet50(input_shape,num_classes)

#%% 모델 요약

model.summary()


#image_path="/content/drive/MyDrive/L_TL2/cat/eye/normal"

#윈도우에서 시행시에
image_path="C:/Users/VSA/Desktop/L_TL2/cat/eye/normal"

#%% 이미지 정상 파일을 
categories=["a","b","c","d","e","f"]



X=[]
Y = []

for idx, cate in enumerate(categories):
    label = [0 for _ in range(num_classes)]
    label[idx] = 1
    image_dir = image_path + '/' + cate + '/Y/'

    for top, dir, files in os.walk(image_dir):
        i = 0
        for filename in files:
            if filename[-4:] != 'json':
                i += 1
                img = cv2.imread(image_dir + filename, 1)
                img = cv2.resize(img, (200, 200))
                img = cv2.copyMakeBorder(img, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=0)
                X.append(img)
                Y.append(label)
 #               X.append(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
 #               Y.append(label)
                if i == 1000:
                    break
X=np.array(X)
X=X.astype(np.float32)/255.0
Y=np.array(Y)

x_train, x_test, y_train, y_test = train_test_split(X, Y)


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

model.compile(optimizer=Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
with tf.device('/GPU:0'):
    history=model.fit(x_train,y_train,epochs=80,batch_size=16)

#model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

#with tf.device('/GPU:0'):
#    history = model.fit(train_generator, epochs=100, steps_per_epoch=len(train_generator), validation_data=(x_test, y_test))



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

