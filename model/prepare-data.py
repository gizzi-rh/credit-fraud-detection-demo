import os
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns
import tf2onnx
from sklearn.linear_model import LogisticRegression
import keras


def splitDataset()
    # Load the CSV data which we will use to train the model.
    # It contains the following fields:
    #   distancefromhome - The distance from home where the transaction happened.
    #   distancefromlast_transaction - The distance from last transaction happened.
    #   ratiotomedianpurchaseprice - Ratio of purchased price compared to median purchase price.
    #   repeat_retailer - If it's from a retailer that already has been purchased from before.
    #   used_chip - If the (credit card) chip was used.
    #   usedpinnumber - If the PIN number was used.
    #   online_order - If it was an online order.
    #   fraud - If the transaction is fraudulent.

    Data = pd.read_csv('../data/card_transdata.csv')
    Data.head()

    # Set the input (X) and output (Y) data. 
    # The only output data we have is if it's fraudulent or not, and all other fields go as inputs to the model.

    X = Data.drop(columns = ['fraud'])
    y = Data['fraud']

    # Split the data into training and testing sets so we have something to test the trained model with.

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, stratify = y)

    X_train, X_val, y_train, y_val = train_test_split(X_train,y_train, test_size = 0.2, stratify = y_train)

    # Scale the data to remove mean and have unit variance. This means that the data will be between -1 and 1, which makes it a lot easier for the model to learn than random potentially large values.
    # It is important to only fit the scaler to the training data, otherwise you are leaking information about the global distribution of variables (which is influenced by the test set) into the training set.

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    # Since the dataset is unbalanced (it has many more non-fraud transactions than fraudulent ones), we set a class weight to weight the few fraudulent transactions higher than the many non-fraud transactions.

    class_weights = class_weight.compute_class_weight('balanced',classes = np.unique(y_train),y = y_train)
    class_weights = {i : class_weights[i] for i in range(len(class_weights))}

    

if __name__ == '__main__':
    splitDataset()