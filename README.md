# Cheetah vs Tiger CNN Classifier

## Project Overview
This project implements a Convolutional Neural Network (CNN) from scratch to classify images of cheetahs and tigers. The model is trained on a custom dataset of labeled images and includes data augmentation techniques such as random flips, rotations, zooms, and brightness adjustments to improve generalization.

The CNN architecture consists of multiple convolutional layers with ReLU activation, max pooling layers, and fully connected dense layers. Dropout is used to reduce overfitting. The model is compiled with the Adam optimizer and binary crossentropy loss function.

## Performance
The trained model achieves an **accuracy of 97%** on the validation set, demonstrating strong performance in distinguishing between cheetahs and tigers.

## Usage
1. Place your test images in the project folder.  
2. Use the provided prediction code to classify a new image:  

```python
from tensorflow.keras.preprocessing import image
import tensorflow as tf

# Load and preprocess the image
img_path = "some.jpg"  # replace with your image file
img = image.load_img(img_path, target_size=(400,400))
x = image.img_to_array(img)
x = tf.expand_dims(x,0)/255.0

# Make prediction
p = model.predict(x)[0][0]
print("Tiger" if p>0.5 else "Cheetah", p)

```
Requirements

Python 3.12+

TensorFlow 2.x (includes Keras)
