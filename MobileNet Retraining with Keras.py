
# coding: utf-8

# In[1]:


# Importing all necessary packages from Keras Applications API 
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from sklearn.metrics import classification_report
from keras.callbacks import EarlyStopping, ModelCheckpoint
from PIL import Image


# In[2]:


import sys
from PIL import Image
sys.modules['Image'] = Image 


# In[6]:


# Parameters 
set_seed = 42
num_classes = 6
batch_size = 16
epochs = 10
patience_epochs = 2
train_val_dir = './split_smoking_images/train/'
test_dir = './split_smoking_images/test/'


# In[7]:


# Create the base pre-trained model, without top dense layers
base_model = MobileNetV2(weights='imagenet', include_top=False)
# We can see all the layers: 
#base_model.summary()


# In[8]:


x = base_model.output
# Add a global spatial average pooling layer to reduce dimensionality.
x = GlobalAveragePooling2D()(x)
# Add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# And a logistic layer of width num_classes 
predictions = Dense(num_classes, activation='softmax')(x)
# Complete model using Model object
model = Model(inputs=base_model.input, outputs=predictions)


# In[9]:


# first: train only the top layers (randomly initialized)
# i.e. freeze all convolutional MobileNetV2 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')


# In[10]:


train_val_datagen = ImageDataGenerator(rotation_range=20, zoom_range=0.15, rescale=1./255,
                         width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
                         horizontal_flip=True, fill_mode="nearest",
                         validation_split=0.2) # set validation split

train_generator = train_val_datagen.flow_from_directory(
    train_val_dir, batch_size=batch_size, shuffle=True, seed=set_seed,
    subset='training') # set as training data to seperate from validation!

val_generator = train_val_datagen.flow_from_directory(
    train_val_dir, batch_size=batch_size, shuffle=True, seed=set_seed,
    subset='validation') # set as validation data to seperate from training!

print(train_generator.class_indices)


# In[11]:


# Callbacks 
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=patience_epochs)
checkpoint_callback = ModelCheckpoint('mobilenetv2'+'.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')


# In[1]:


# Training
model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = val_generator, 
    validation_steps = val_generator.samples // batch_size,
    epochs = epochs,
    callbacks=[early_stopping_callback, checkpoint_callback], verbose=1
)


# In[ ]:


# Make a final data generator, for evaluating.
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = train_val_datagen.flow_from_directory(
    test_dir, batch_size=batch_size, shuffle=False) 

loss, acc = model.evaluate_generator(test_generator, verbose=1)

