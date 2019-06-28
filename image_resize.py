import tensorflow as tf
from PIL import Image
import os, os.path
import skimage
from skimage import data, transform, io
import numpy as np
import matplotlib.pyplot as plt


train_data_directory = "./ourDataset/training"
test_data_directory = "./ourDataset/test"


def load_data(data_directory):
    imag = []
    valid_images = ".png"
    for f in os.listdir(data_directory):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        imag.append(skimage.data.imread(os.path.join(data_directory, f)))
    return imag


images = load_data(train_data_directory)

imagearr = np.array(images)

print(imagearr.size)
print(imagearr.ndim)


pokemon = [0, 1, 2, 3]
for i in range(len(pokemon)):
    plt.subplot(2, 2, i+1)
    plt.axis('off')
    plt.imshow(images[pokemon[i]])
    plt.subplots_adjust(wspace=0.5)
plt.show()

# Rescale the images in the `images` array
images28 = [transform.resize(image, (28, 28)) for image in images]
for i in range(len(pokemon)):
    plt.subplot(2, 2, i+1)
    plt.axis('off')
    plt.imshow(images28[pokemon[i]])
    plt.subplots_adjust(wspace=0.5)
plt.show()

for i in range(len(images28)):
    skimage.io.imsave("./save/pokemon{0}.png".format(i), images28[i])
