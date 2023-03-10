# coding: utf-8

# In[ ]:
import os

import tensorflow as tf
# sess = tf.Session()
import keras
# from keras.engine.saving import load_model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Activation, Dropout, Flatten

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

import numpy as np

#------------------------------
# sess = tf.Session()
# keras.backend.set_session(sess)
#------------------------------
#variables
num_classes =37
batch_size = 40
epochs = 10
#------------------------------

import os, cv2, keras
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow import keras
from keras.utils import np_utils
#from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
#from keras.engine.saving import load_model
from tensorflow.keras.models import load_model
from keras import utils as np_utils
from tensorflow.keras.utils import to_categorical
# manipulate with numpy,load with panda
import numpy as np
# import pandas as pd

# data visualization
import cv2
import matplotlib
import matplotlib.pyplot as plt
# import seaborn as sns

# get_ipython().run_line_magic('matplotlib', 'inline')


# Data Import
def read_dataset():
    data_list = []
    label_list = []
    i=0
    my_list = os.listdir(r'/Users/mariyam/Documents/sign language recognition - Cop/dataSet/trainingData/')
    for pa in my_list:

        print(pa,"==================",i)
        for root, dirs, files in os.walk(r'/Users/mariyam/Documents/sign language recognition - Cop/dataSet/trainingData//' + pa):

         for f in files:
            file_path = os.path.join(r'/Users/mariyam/Documents/sign language recognition - Cop/dataSet/trainingData//'+pa, f)
            # print(file_path)
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            res = cv2.resize(img, (48, 48), interpolation=cv2.INTER_CUBIC)
            data_list.append(res)
            # label = dirPath.split('/')[-1]
            label = i
            label_list.append(label)
        i=i+1

    return (np.asarray(data_list, dtype=np.float32), np.asarray(label_list))

def read_dataset1(path):
    data_list = []
    label_list = []

    file_path = os.path.join(path)
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    res = cv2.resize(img, (48, 48), interpolation=cv2.INTER_CUBIC)
    data_list.append(res)
    # label = dirPath.split('/')[-1]

            # label_list.remove("./training")
    return (np.asarray(data_list, dtype=np.float32))

from sklearn.model_selection import train_test_split
# load dataset
x_dataset, y_dataset = read_dataset()
X_train, X_test, y_train, y_test = train_test_split(x_dataset, y_dataset, test_size=0.2, random_state=0)

y_train1=[]
for i in y_train:
    emotion = tf.keras.utils.to_categorical(i, num_classes)
    #keras.utils.to_categorical
    print(i,emotion)
    y_train1.append(emotion)

y_train=y_train1

x_train = np.array(X_train, 'float32')
y_train = np.array(y_train, 'float32')
x_test = np.array(X_test, 'float32')
y_test = np.array(y_test, 'float32')

x_train /= 255  # normalize inputs between [0, 1]
x_test /= 255
print("x_train.shape",x_train.shape)
x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
x_train = x_train.astype('float32')
x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)
x_test = x_test.astype('float32')

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
# ------------------------------

