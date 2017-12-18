'''
author: Xiaoli Shang
web: https://github.com/iamacat9/web-prediction
license: CC0 1.0 Universal (free to use, no warranties)

This code sample is a bare bones demo of the following workflow:

1. load a training dataset
2. train a classifier with your favorite algorithm
  (optional) do this in parallel when the training set is large
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
    mdl = LogisticRegression(penalty='l2')
    mdl.fit(tr_data, tr_label)
    
    return mdl

# partition the training set into n_child chunks and train n_child models in parallel
def train_model_parallel(tr_data, tr_label, n_child):
    
    from multiprocessing import Process, Queue
    
    # set number of jobs
    try:
        n_child = max(2, int(n_child))
    except ValueError:
        n_child = 2
    print 'parallelization with %i jobs' % n_child
    
    N = tr_data.shape[0] # number of training instances
    chunk_size = int(numpy.ceil(N/n_child))
    idx = numpy.random.permutation(N) # random permutation to create random samples
    
    result_q = Queue() # collect results
    process_list = []
    for i in xrange(n_child):
        n1 = i * chunk_size
        n2 = min((i+1) * chunk_size, N)
        # pass in a chunk of data and their labels
        p = Process(target=train_model_child, \
                    args=(result_q, tr_data[idx[n1:n2],:], tr_label[idx[n1:n2]]))
        process_list.append(p)
    
    # dispatch and sync
    for p in process_list:
        p.start()
    for p in process_list:
        p.join()
    
    # ensemble n_child different models
    mdl = model_ensemble(result_q)
    
    return mdl

# train a model on each chunk of data
def train_model_child(result_q, tr_data, tr_label):
    
    mdl_c = train_model(tr_data, tr_label) # train the model using existing function
    result_q.put(mdl_c) # put the trained model into the queue

# ensemble all models in result_q into one
def model_ensemble(result_q):
    
    # simple model averaging for logistic regression
    
    mdl = result_q.get()
    cnt = 1
    while not result_q.empty():
        mdl_c = result_q.get()
        mdl.coef_ += mdl_c.coef_
        mdl.intercept_ += mdl_c.intercept_
        cnt += 1
    
    mdl.coef_ /= cnt
    mdl.intercept_ /= cnt
    
    return mdl    

if __name__ == '__main__':
    
    # load data
    if os.path.isfile('data.csv'):
        tr_data, tr_label, fea_list = load_dataset('data.csv')
    else:
        tr_data, tr_label, fea_list = generate_dataset()
        
    # train model
    #mdl = train_model(tr_data, tr_label)
    mdl = train_model_parallel(tr_data, tr_label, n_child=2)
    
    # define url structure
    urls = (
        '/', 'index',
        '/pred', 'pred'
    )
    
    # start web application
    app = web.application(urls, globals())
    app.run()

