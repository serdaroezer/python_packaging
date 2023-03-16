import pickle
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import Ridge
import numpy as np
from os import path
from .helper import read_data, mean_normalize, split_test_train, drop_columns, remove_outlier_with_boxplot


def train_regression_model(data_path, split_rate, label_name, model_name, dropped_columns):
    df = read_data(data_path)
    df = remove_outlier_with_boxplot(df)

    # get label from dataframe
    y = df[label_name]

    # remove label and some other columns from training data
    dropped_columns.append(label_name)
    x = drop_columns(df, dropped_columns)

    x = mean_normalize(x)

    x_train, x_test, y_train, y_test = split_test_train(split_rate, x, y)

    ridge = Ridge( alpha=1, solver='saga')
    ridge.fit(x_train, y_train)

    y_pred = ridge.predict(x_test)

    pickle.dump(ridge, open(model_name + ".pkl", 'wb'))

    return apply_metric(y_test, y_pred)


def predict(data_path, label_name, model_name, dropped_columns):
    df = read_data(data_path)

    # get label from dataframe
    y_test = df[label_name]

    # remove label and some other columns from training data
    dropped_columns.append(label_name)
    x_test = drop_columns(df, dropped_columns)

    x_test = mean_normalize(x_test)

    model = pickle.load(open(path.join(model_name + '.pkl'), mode='rb'))

    if model is None:
        print("failed to load model")
        return

    y_pred = model.predict(x_test)
    return apply_metric(y_test, y_pred)


def apply_metric(y_test, y_pred):
    residual = pd.DataFrame({'Y_Test': y_test, 'Y_Pred': y_pred, 'Residuals': (y_test - y_pred)}).head(5)
    MSE = metrics.mean_squared_error(y_test, y_pred)
    MAE = metrics.mean_absolute_error(y_test, y_pred)
    RMSE = np.sqrt(MSE)

    print(residual)
    print('MAE: ', MAE)
