#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import keras
import numpy as np 
import matplotlib.pyplot as plt
import os


# In[2]:


tf.config.list_physical_devices()


# In[3]:


train_d = keras.preprocessing.image.ImageDataGenerator(zoom_range=0.5, shear_range=0.3, horizontal_flip=True, preprocessing_function=keras.applications.vgg19.preprocess_input)


# In[4]:


valid_d = keras.preprocessing.image.ImageDataGenerator( preprocessing_function=keras.applications.vgg19.preprocess_input)


# In[5]:


train_data = train_d.flow_from_directory("C:\\Users\\Natsu\\Downloads\\archive\\New Plant Diseases Dataset(Augmented)\\New Plant Diseases Dataset(Augmented)\\train", batch_size=32, target_size=(255,255))


# In[6]:


valid_data = valid_d.flow_from_directory('C:\\Users\\Natsu\\Downloads\\archive\\New Plant Diseases Dataset(Augmented)\\New Plant Diseases Dataset(Augmented)\\valid', batch_size=32, target_size=(255,255))


# In[7]:


img, label = train_data.next()


# In[8]:


img.shape


# In[9]:


def plot_image(img, label):
    for img, l in zip(img, label):
        plt.figure(figsize=(10,10))
        plt.imshow(img/255)
        plt.show()
        plt.axis('off')


# In[10]:


plot_image(img[:5], label[:5])


# In[11]:


from keras.applications.vgg19 import VGG19
base_model = VGG19(input_shape=(255, 255, 3), include_top=False)


# In[12]:


for layer in base_model.layers:
    layer.trainable = False 


# In[13]:


base_model.summary()


# In[14]:


x = keras.layers.Flatten()(base_model.output)
x = keras.layers.Dense(units=38, activation='softmax')(x)


# In[15]:


model = keras.models.Model(base_model.input, x)


# In[16]:


model.summary()


# In[17]:


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[22]:


# Early Stopping and callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping
es = EarlyStopping(monitor='val_accuracy', min_delta=0.01, patience=3, verbose=1)

#model checkpoint
mc = ModelCheckpoint(filepath='best_model.h5', monitor='val_accuracy', patience=3, min_delta=0.01, verbose=1, save_best_only=True)
cb = [es, mc]


# In[24]:


his = model.fit(train_data, epochs=10, callbacks=cb, verbose=1,validation_data=valid_data, batch_size=100 )


# In[27]:


h = his.history
h.keys()


# In[30]:


plt.plot(h['accuracy'])
plt.plot(h['val_accuracy'])
plt.legend(['accuracy', 'val_accuracy'])
plt.title('Accuracy')
plt.show()


# In[31]:


from keras.models import load_model
model = load_model('best_model.h5')


# In[36]:


acc = model.evaluate_generator(valid_data)[1]


# In[37]:


acc


# In[2]:


import pathlib
test_dir = pathlib.Path('C:\\Users\\Natsu\\Downloads\\archive\\test')
test_dir


# In[ ]:


from keras.preprocessing.image import ImageDataGenerator
test_gen = ImageDataGenerator(1./255)

#test_d = 'C:\\Users\\Natsu\\Downloads\\archive\\New Plant Diseases Dataset(Augmented)\\New Plant Diseases Dataset(Augmented)\\test'
test_data = test_gen.flow_from_directory(test_dir, batch_size=100, target_size=(255,255))


# In[ ]:




