#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import librosa
import os
##if you need any imports you can do that here. 


# In[1]:


get_ipython().system('unzip "recordings.zip"')


# In[4]:


#read the all file names in the recordings folder given by us
#(if you get entire path, it is very useful in future)
#save those files names as list in "all_files"

common_path = "recordings/"

temp = os.listdir(common_path)
all_files = [common_path + i for i in temp]
all_files[:5]


# In[5]:


#Create a dataframe(name=df_audio) with two columns(path, label).   
#You can get the label from the first letter of name.  
#Eg: 0_jackson_0 --> 0  
#0_jackson_43 --> 0
df_audio = pd.DataFrame()

df_audio["path"] = all_files
df_audio["label"] = [int(i[len(common_path):len(common_path) + 1]) for i in all_files]

df_audio


# In[6]:


sample_rate = 22050
def load_wav(x, get_duration=True):
    '''This return the array values of audio with sampling rate of 22050 and Duration'''
    #loading the wav file with sampling rate of 22050
    samples, sample_rate = librosa.load(x, sr=22050)
    if get_duration:
        duration = librosa.get_duration(samples, sample_rate)
        return [samples, duration]
    else:
        return samples


# In[7]:


## generating augmented data. 
def generate_augmented_data(file_path):
    augmented_data = []
    samples = load_wav(file_path,get_duration=False)
    for time_value in [0.7, 1, 1.3]:
        for pitch_value in [-1, 0, 1]:
            time_stretch_data = librosa.effects.time_stretch(samples, rate=time_value)
            final_data = librosa.effects.pitch_shift(time_stretch_data, sr=sample_rate, n_steps=pitch_value)
            augmented_data.append(final_data)
    return augmented_data


# In[8]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_audio.path, df_audio.label, test_size=0.30, random_state=33 , stratify=df_audio.label)


# In[9]:


train_df = pd.DataFrame()
temp = X_train.reset_index()
temp2 = y_train.reset_index()

train_df["path"] = temp["path"]
train_df["label"] = temp2["label"]
train_df 


# In[10]:


def return_augmented_df(data , label):
    var_list = []
    for i in tqdm(data["path"]):
        
        lr = generate_augmented_data(i)
        for j in lr:
            var_list.append(j)
    label = [label]*(data.shape[0]*9)
    return var_list , label


# In[11]:


#For some reason for loop is not working.

from tqdm import tqdm
lr0 , label0 =  return_augmented_df(train_df[train_df.label == 0] , 0)
lr1 , label1 =  return_augmented_df(train_df[train_df.label == 1] , 1)
lr2 , label2 =  return_augmented_df(train_df[train_df.label == 2] , 2)
lr3 , label3 =  return_augmented_df(train_df[train_df.label == 3] , 3)
lr4 , label4 =  return_augmented_df(train_df[train_df.label == 4] , 4)
lr5 , label5 =  return_augmented_df(train_df[train_df.label == 5] , 5)
lr6 , label6 =  return_augmented_df(train_df[train_df.label == 6] , 6)
lr7 , label7 =  return_augmented_df(train_df[train_df.label == 7] , 7)
lr8 , label8 =  return_augmented_df(train_df[train_df.label == 8] , 8)
lr9 , label9 =  return_augmented_df(train_df[train_df.label == 9] , 9)


# In[12]:


lr = lr0 + lr1 + lr2 + lr3 + lr4 + lr5 + lr6 + lr7 + lr8 + lr9
label = label0 + label1 + label2 + label3 + label4 + label5 + label6 + label7 + label8 + label9

train_df = pd.DataFrame(columns = ["raw_data" , "label"])
train_df["raw_data"] = lr
train_df["label"] = label

#My split is 30%-----> Hope thats ok.(As my RAM GB is exceeding.)
train_df


# In[13]:


test_dict = {"raw_data" : [] , "duration" : []}
for i in tqdm(X_test):
    try:
        sample , duration = load_wav(i)
        test_dict["raw_data"].append(sample)
        test_dict["duration"].append(duration)
    except:
        continue  


# In[14]:


max_length  = 17640 
X_train_processed = train_df
X_test_processed = pd.DataFrame(test_dict)


# In[15]:


#Cleaning

for i in lr:
    del i

del lr

for i in label:
    del i

del label
del test_dict


# In[16]:


def return_pad(x):
    if len(x) > max_length:
        x = x[:max_length]
    else:
        no_of_pad = max_length - len(x)
        x = list(x) + [0]*no_of_pad
    return x

def return_mask(x):
    if len(x) > max_length:
        x = [1]*max_length
    else:
        no_of_pad =  max_length - len(x)
        x = [1]*len(x) + [0]*no_of_pad
    return x



# In[17]:


X_train_pad_seq = X_train_processed["raw_data"].apply(return_pad)
X_train_mask = X_train_processed["raw_data"].apply(return_mask)


# In[18]:


X_test_pad_seq = X_test_processed["raw_data"].apply(return_pad)
X_test_mask = X_test_processed["raw_data"].apply(return_mask)


# In[19]:


print("Pad-Seq")
print(X_train_pad_seq.shape , X_test_pad_seq.shape)
print("Masks")
print(X_train_mask.shape , X_test_mask.shape)


# In[20]:


def expand_object(array):
    df = pd.DataFrame()
    for i in tqdm(range(max_length)):
        df[i] = [k[i] for k in array]
    
    return df.to_numpy()

X_train_pad_seq = expand_object(X_train_pad_seq)
X_test_pad_seq = expand_object(X_test_pad_seq)

X_train_mask = expand_object(X_train_mask)
X_test_mask = expand_object(X_test_mask)

print("Pad-Seq")
print(X_train_pad_seq.shape , X_test_pad_seq.shape)
print("Masks")
print(X_train_mask.shape , X_test_mask.shape)


# In[21]:


X_train_mask = X_train_mask.astype("bool")
X_test_mask = X_test_mask.astype("bool")
print("Masks")
print(X_train_mask.shape , X_test_mask.shape)


# In[22]:


from tensorflow.keras.layers import Input, LSTM, Dense,Flatten
from tensorflow.keras.models import Model
import tensorflow as tf


# In[23]:


from sklearn.metrics import f1_score
def fi_micro_temp(y_true, y_pred):
    y_true = tf.argmax(y_true, axis=1)
    y_pred = tf.argmax(y_pred, axis=1)
    return f1_score(y_true, y_pred, average='micro')

def f1_micro(y_true, y_pred):
    
    return tf.py_function(fi_micro_temp, (y_true, y_pred), tf.double)


# In[30]:


max_length = 17640
input_layer = Input(shape = (max_length ,1))
input_mask = Input(shape = (max_length,),dtype = "bool")

X = LSTM(25)
output = X(inputs = input_layer ,mask = input_mask)

output = Dense(50 , activation = "relu" ,kernel_initializer='glorot_normal')(output)

output = Dense(10 , activation = "relu" ,kernel_initializer='glorot_normal')(output)

model = Model([input_layer,input_mask] , output)

model.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.Adam(lr=0.001),metrics=[f1_micro])

model.summary()


# In[25]:


train = [X_train_pad_seq , X_train_mask]
test = [X_test_pad_seq , X_test_mask]


# In[26]:


y_train = train_df["label"]


# In[27]:


y_train = tf.keras.utils.to_categorical(y_train , num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test , num_classes=10)


# In[28]:


print(y_train.shape , y_test.shape)


# In[31]:


model.fit(train, y_train, validation_data=(test, y_test),epochs=1)


# In[32]:


def convert_to_spectrogram(raw_data):
    '''converting to spectrogram'''
    spectrum = librosa.feature.melspectrogram(y=raw_data, sr=sample_rate, n_mels=64)
    logmel_spectrum = librosa.power_to_db(S=spectrum, ref=np.max)
    return logmel_spectrum


# In[33]:


def return_spec_dataframe(x):
    var_list = []
    for i in tqdm(x):
        var_list.append(convert_to_spectrogram(i))
    return np.array(var_list)



X_train_spectrogram = return_spec_dataframe(X_train_pad_seq)
X_test_spectrogram = return_spec_dataframe(X_test_pad_seq)

print()
print("Train:")
print(X_train_spectrogram.shape)
print("X_test:")
print(X_test_spectrogram.shape)


# In[39]:


input_layer = Input(shape = (64 ,35))

X = LSTM(800 , return_sequences=True)(input_layer)
X = tf.keras.layers.GlobalAveragePooling1D()(X)

output = Dense(10 , activation = "softmax" ,kernel_initializer='glorot_normal')(X)

model = Model(input_layer , output)

model.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.Adam(lr=0.001),metrics=[f1_micro])

model.summary()


# In[37]:


import datetime
log_dir="logs1\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,histogram_freq=1, write_graph=True,write_grads=True)


# In[40]:


model.fit(X_train_spectrogram, y_train, validation_data=(X_test_spectrogram, y_test),epochs=20,callbacks = [tensorboard_callback])


# In[41]:


get_ipython().run_line_magic('load_ext', 'tensorboard')


# In[42]:


get_ipython().run_line_magic('tensorboard', '--logdir  .')

