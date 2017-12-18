webprediction
========

This code sample is a bare bones demo of the following workflow:

1. load a training dataset from a local file
2. train a classifier with your favorite machine learning algorithm
3. launch a RESTful web service to apply the trained model to future test data

This is a simple way of integrating a prediction module into a larger web-based analytics platform

USAGE: python demo.py \<ip\>:\<port\>

Then open \<ip\>:\<port\> in your web browser

Tested on: Python 2.7, sklearn 0.13