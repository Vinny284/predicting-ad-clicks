# Predicting Ad Clicks

## Goals

The goal of this project is to build a model which predicts if a user has clicked on an advertisment based on the websites they have visted and to gain insight into which types of websites adverts should be placed on.
The file data2.csv contains a sample of 10,000 rows of user activity, showing the number of ad views per site per user. Also joined to this data, is whether or not the user clicked on the ad, shown as 1 for a click and 0 otherwise.

This is a modified file where the websites are grouped by type, the user ID's have been removed and the values are edited as I cannot upload the original dataset.
Since the file has already been modified, the data is already clean, i.e. there are no NULL values and no input errors like extreme values, but there are still outliers, which are dealt with in Outliers.ipynb.

## Key Findings

There are around 7 times more no-clicks than clicks.
The sites most likely to get clicks are blog sites and opinion sites.
The sites least likely to get clicks are document sites, social media measurment sites and news sites.
Through some experimentation, I found that when the number of impressions in the test data were capped at 5, the predicted number of clicks did not change significantly on average. This is a cutoff point where further views won't affect the number of clicks by much.

## The Model

The data is unbalanced so I am oversampling the smaller class in the training to counter this. I am using a 80/20 train/test split for the data and I have removed outliers. The chosen model is a random forest classifier, since the data is not neccessarily linear and it can be easily tuned to under or over fit to the data. I am using precision, recall and f1 score as my oerformance metrics. The model when run outputs:

TRAINING METRICS: 

Confusion matrix:

[[7137  247]
 [ 617 2497]]
 
Recall: 80.18625561978163

Precision: 90.99854227405248

F1 Score: 85.25093888699216

TEST METRICS: 

Confusion matrix:

[[783  37]
 [ 47 114]]
 
Recall: 70.80745341614907

Precision: 75.49668874172185

F1 Score: 73.07692307692308
