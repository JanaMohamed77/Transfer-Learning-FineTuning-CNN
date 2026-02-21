# Transfer-Learning-FineTuning-CNN
Project Overview

This project demonstrates how to apply Transfer Learning and Fine-Tuning using a pretrained CNN model to perform binary image classification (Cats vs Dogs).

We used:

MobileNetV2 pretrained on ImageNet

TensorFlow & Keras

Data Augmentation for regularization

 Steps
1- Load Pretrained Model

We used MobileNetV2 without the top classification layer.

2- Transfer Learning

All convolutional layers were frozen and only the new classifier was trained.

3- Fine-Tuning

The last 20 layers were unfrozen and retrained with a low learning rate.

 Results

Fine-Tuning improved validation accuracy and reduced overfitting.

Technologies

TensorFlow

Keras

MobileNetV2
