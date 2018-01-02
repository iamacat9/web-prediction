'''
author: Xiaoli Shang
web: https://github.com/iamacat9/web-prediction
license: CC0 1.0 Universal (free to use, no warranties)

This code sample is a bare bones demo of the following workflow:

1. load a training dataset
2. train a classifier with your favorite algorithm
3. launch a RESTful web service to apply the trained model to future test data

This is a simple way of integrating a prediction module into a larger web-based analytics platform

USAGE: "python demo.py <ip>:<port>", then open "<ip>:<port>" in your web browser

Tested on: Python 2.7, sklearn 0.13+
'''

import csv
import web
import numpy
import json
import os  

class index:
    
    # use the default landing page to provide usage information
    def GET(self):
        
        mytext = '<p>Usage: &lt;ip&gt;:&lt;port&gt;/pred?A=1.2&B=0.8</p>'
        mytext += '<p>List of feature names: ' + str(fea_list) + '</p>'
        return mytext

class pred:
    
    # make prediction on record (1xD)
    def prediction(self, record):
        
        score = mdl.predict_proba(record)[0][1]
        score = '%.4f' % score 
        aux_info = '' # put additional information in here
        
        return score, aux_info
    
    # take web input, make prediction, and dump result
    def GET(self):
        
        # initialize record from web input
        user_data = web.input()
        record = numpy.zeros((1,len(fea_list)))
        for key in user_data.keys():
            if key in fea_list:
                try:
                    record[0, fea_list.index(key)] = float(user_data[key])
                except ValueError:
                    return json.dumps(['', 'invalid feature value!'])
        
        score, aux_info = self.prediction(record)
        
        return json.dumps([score, aux_info])

# load training data from a csv file
def load_dataset(filename):
    
    '''
    Expected Format
    -----------------
    A,B,C,D,label
    5.1,3.5,1.4,0.2,0
    4.9,3.0,1.4,0.2,0
    5.7,2.9,4.2,1.3,1
    6.2,2.9,4.3,1.3,1
    -----------------
    '''
    
    # extract the list of feature names
    with open(filename,'rb') as f:
        reader = csv.reader(f)
        fea_list = reader.next()
    fea_list = fea_list[:-1] # drop the last item 'label'
    
    # extract data
    data_col = tuple(range(len(fea_list))) # skip last column
    tr_data = numpy.genfromtxt(filename, delimiter=',', skiprows=1, usecols=data_col)
    
    # extract label
    label_col = (len(fea_list),) # last column only
    tr_label = numpy.genfromtxt(filename, delimiter=',', skiprows=1, usecols=label_col)

    return tr_data, tr_label, fea_list

# generate a random dataset
def generate_dataset():
    
    print 'Warning: we are using randomly generated data'
    tr_data = numpy.random.rand(1000, 4)
    tr_label = numpy.random.randint(2, size=1000)
    fea_list = ['A','B','C','D']
    
    return tr_data, tr_label, fea_list

# choose your favorite classifier to train the model
def train_model(tr_data, tr_label):
    
    # here we use logistic regression from the sklearn library
    from sklearn.linear_model import LogisticRegression
    mdl = LogisticRegression()
    mdl.fit(tr_data, tr_label)
    
    return mdl


if __name__ == '__main__':
    
    # load data
    if os.path.isfile('data.csv'):
        tr_data, tr_label, fea_list = load_dataset('data.csv')
    else:
        tr_data, tr_label, fea_list = generate_dataset()
        
    # train model
    mdl = train_model(tr_data, tr_label)
    
    # define url structure
    urls = (
        '/', 'index',
        '/pred', 'pred'
    )
    
    # start web application
    app = web.application(urls, globals())
    app.run()

