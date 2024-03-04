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
import mlflow
from sklearn.linear_model import LogisticRegression
import keras
import onnx


# Train the model.
# We wrap the training with an mlflow wrapper to signify that this is an experiment run.
# We also define a few more metrics at the very bottom to track the confusion matrix in MLFlow.

def prepareMlFlow():
    MLFLOW_ROUTE = os.getenv("MLFLOW_ROUTE")
    mlflow.set_tracking_uri(MLFLOW_ROUTE)
    mlflow.set_experiment("DNN-credit-card-fraud")
    mlflow.tensorflow.autolog(registered_model_name="DNN-credit-card-fraud")

def train():
    prepareMlFlow()

    with mlflow.start_run():
        epochs = 2
        history = model.fit(X_train, y_train, epochs=epochs, \
                            validation_data=(scaler.transform(X_val),y_val), \
                            verbose = True, class_weight = class_weights)

        y_pred_temp = model.predict(scaler.transform(X_test)) 

        threshold = 0.995

        y_pred = np.where(y_pred_temp > threshold, 1,0)
        c_matrix = confusion_matrix(y_test,y_pred)
        ax = sns.heatmap(c_matrix, annot=True, cbar=False, cmap='Blues')
        ax.set_xlabel("Prediction")
        ax.set_ylabel("Actual")
        ax.set_title('Confusion Matrix')
        plt.show()

        t_n, f_p, f_n, t_p = c_matrix.ravel()
        mlflow.log_metric("tn", t_n)
        mlflow.log_metric("fp", f_p)
        mlflow.log_metric("fn", f_n)
        mlflow.log_metric("tp", t_p)

        model_proto, _ = tf2onnx.convert.from_keras(model)
        mlflow.onnx.log_model(model_proto, "models")

        saveOnnxModel(model_proto, "fraud.onnx")


def saveOnnxModel(model_proto: ModelProto, file_name: str):
    onnx.save(model_proto, file_name)


if __name__ == '__main__':
    train()