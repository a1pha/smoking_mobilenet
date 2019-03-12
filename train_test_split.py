import os
import shutil
import numpy as np
source1 = "./smoking_images/"
dest1 = "./split_smoking_images/"
file = os.listdir(source1)
for img_class in file:
    os.makedirs(dest1 + 'test/' + img_class + '/')
    os.makedirs(dest1 + 'train/' + img_class + '/')
    if not os.path.isdir(source1+str(img_class)):
        continue
    images = os.listdir(source1+str(img_class))
    for image in images:
        if np.random.rand(1) < 0.2:
            shutil.copy(source1 + img_class + '/' + image,
             dest1 + 'test/' + img_class + '/' + image)
        else:
            shutil.copy(source1 + img_class + '/' + image,
             dest1 + 'train/' + img_class + '/' + image)
             
