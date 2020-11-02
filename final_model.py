import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import *
import sys

TEST_MODEL = True # set to true to split the data into train-test data for validation
DISPLAY_TRAIN_METRICS = True # Display confusion matrix, precion, recall, f1 score of training set
DISPLAY_TEST_METRICS = True # Display confusion matrix, precion, recall, f1 score of test set if TEST_MODEL=True


try:
    df = pd.read_csv('data3.csv')
except FileNotFoundError:
    print('File not in directory.')
    sys.exit(0)
    

'''
PREPARING TRAINING DATA

The data is very unbalanced, there are less clicks than no-clicks.
To counter this, I am oversampling the smaller class (clicks).

The amount of oversampling determines if the model prioritises
precision or recall.

'''

# split data by class nc = no click, c = click 
nc = df.loc[df['Click'] == 0]
c = df.loc[df['Click'] == 1]


TRAIN_SAMPLE_FRAC = 0.9 # fraction of dataset used for training if TEST_MODEL=True, between 0 and 1

if TEST_MODEL:
    # sample training data
    nc_train = nc.sample(frac=TRAIN_SAMPLE_FRAC, random_state=0)
    c_train = c.sample(frac=TRAIN_SAMPLE_FRAC, random_state=0)
else:
    # if TEST_MODEL = False, then train on the entire test
    nc_train = nc
    c_train = c
   

# balance training data by oversampling the smaller class and recombine into 1 data frame
# the over_sampling_multiplier can be adjusted to prioritise precision or recall
# higher values = more recall, lower values = more precision
over_sampling_multiplier = 3.0
c_train = c_train.sample(frac=over_sampling_multiplier, replace=True)
train = pd.concat([nc_train, c_train])


# create features and labels for training
x = train.iloc[:,1:]
y = train.iloc[:, 0]


if TEST_MODEL:
    # sample test data
    nc_test = nc.drop(nc_train.index)
    c_test = c.drop(c_train.index)
    test = pd.concat([nc_test, c_test])
    
    ## create features and labels for testing
    x_test = test.iloc[:,1:]
    y_test = test.iloc[:, 0]

'''
FEATURE SELECTION


'''
select = SelectFpr(chi2)
x = select.fit_transform(x, y)
x_test = select.transform(x_test)

'''
MODEL SELECTION

I am using a random forest classifier because ensemble methods are genrally
good at dealing with unbalaned data.

The max_depth paramter can be tuned for optimal pefromance. The larger the value
the more the model fits to the training data.

'''
# train model
model = RandomForestClassifier(max_depth=10)
model.fit(x, y)
if TEST_MODEL:
    y_pred_test = model.predict(x_test)
    y_pred = model.predict(x)


'''
DISPLAY PEFORMANCE METRICS

Outputs the confusion matrix, precision, recall and f1 score of the train and
test set to the console.

'''
if DISPLAY_TRAIN_METRICS:
    print('TRAINING METRICS: \n')
    cf = confusion_matrix(y, y_pred)
    print('Confusion matrix:')
    print(cf)
    
    recall = 100*cf[1,1]/ (cf[1,0] + cf[1,1]) 
    print('Recall: ' + str(recall))
    
    precision = 100*cf[1,1]/ (cf[0,1] + cf[1,1])
    print('Precision: ' + str(precision))
    
    f1_score = 2*precision*recall/(precision + recall)
    print('F1 Score: ' + str(f1_score) + '\n') 



if DISPLAY_TEST_METRICS and TEST_MODEL:
    print('TEST METRICS: \n')
    cf = confusion_matrix(y_test, y_pred_test)
    print('Confusion matrix:')
    print(cf)
    
    recall = 100*cf[1,1]/ (cf[1,0] + cf[1,1]) 
    print('Recall: ' + str(recall))
    
    precision = 100*cf[1,1]/ (cf[0,1] + cf[1,1])
    print('Precision: ' + str(precision))
    
    f1_score = 2*precision*recall/(precision + recall)
    print('F1 Score: ' + str(f1_score))

    
    
    
    
    
    
    
    