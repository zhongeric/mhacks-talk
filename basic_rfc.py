from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from random import shuffle
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import coremltools
import time

# INITIALIZE HERE
np.random.seed(1)

sample_words = ['how','do','I','do','cpr']

df = pd.read_csv('sample_words.csv')

# Pull labels, drop labels
Y = df['Step']
df = df.drop('Step', 1)

X = df
categories = list(X.columns)
dummy_data = pd.get_dummies(df['Input'])
# d_values = dummy_data.values
# oneD = []
# for value in d_values:
#     print(value)
#     oneD.append(value)
# print(oneD)
# df = df.assign(new = pd.Series(oneD))
# print(df)
#df = df.drop('Input', 1)

print('The shape of our features is:', df.shape)

def getAccuracy(pre,ytest):
    count = 0
    for i in range(len(ytest)):
        if ytest[i]==pre[i]:
            count+=1
    acc = float(count)/len(ytest)
    return acc

#Extract data values from the data frame
feat = df.keys()
feat_labels = df.get_values()

from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(df, Y, test_size = 0.25, random_state = 42)

print('Training Features Shape:', X_train.shape)
print('Training Labels Shape:', Y_train.shape)
print('Testing Features Shape:', X_test.shape)
print('Testing Labels Shape:', Y_test.shape)

clf = RandomForestClassifier()

clf.fit(X_train, Y_train)

pre = clf.predict(X_test)

# Get probability predictions
Y_array = pd.np.array(Y_test)

acc = getAccuracy(pre, Y_array)

# color_input = input("What color is the fruit? ")
# shape_input = input("What shape is the fruit? (sphere or naw) ")
# taste_input = input("Is the fruit sweet or sour? ")
given_pre = clf.predict(np.array([1,0,0]).reshape(1,-1))
print(given_pre)
print(acc)
