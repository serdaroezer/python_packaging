import pickle
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV
import numpy as np
from os import path


def max_min_normalize(X):
    return (X - X.min()) / (X.max() - X.min())


def mean_normalize(X):
    return (X - X.mean()) / X.std()


def read_data(path: str, label: str,dropped_columns):
    df = pd.read_csv(path)
    y = df['Y house price of unit area']
    x = df.drop([label, dropped_columns], axis=1)
    return x, y


def split_test_train(split_rate: int, X, y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=split_rate / 100, random_state=1, shuffle=False)
    return x_train, x_test, y_train, y_test


def polynomial_transform_train_data(polynom_degree: int, x_train):
    poly = PolynomialFeatures(degree=polynom_degree)
    x_train_poly = poly.fit_transform(x_train)
    return x_train_poly


def train_polynomial_model(data_path: str, split_rate: int, polynom_degree: int, label_name: str, model_name: str , dropped_columns):
    x, y = read_data(data_path, label_name, dropped_columns)
    x = mean_normalize(x)

    x_train, x_test, y_train, y_test = split_test_train(split_rate, x, y)

    x_train_poly = polynomial_transform_train_data(polynom_degree, x_train)
    x_test_poly = polynomial_transform_train_data(polynom_degree, x_test)

    lasso_cv = LassoCV(verbose=2, cv=5, max_iter=1000)
    lasso_cv.fit(x_train_poly, y_train)

    model_global = lasso_cv
    y_pred = model_global.predict(x_test_poly)

    pickle.dump(lasso_cv, open(model_name + ".pkl", 'wb'))

    return apply_metric(y_test, y_pred)


def predict(data_path: str, label_name: str, model_name: str, polynom_degree: int):
    x_test, y_test = read_data(data_path, label_name)
    x_test = mean_normalize(x_test)

    x_test_poly = polynomial_transform_train_data(x_test, polynom_degree)

    model = pickle.load(open(path.join(model_name + '.pkl'), mode='rb'))

    if model is None:
        print("failed to load model")
        return

    y_pred = model.predict(x_test_poly)
    return apply_metric(y_test, y_pred)


def apply_metric(y_test, y_pred):
    residual = pd.DataFrame({'Y_Test': y_test, 'Y_Pred': y_pred, 'Residuals': (y_test - y_pred)}).head(5)
    MSE = metrics.mean_squared_error(y_test, y_pred)
    # MAE = metrics.mean_absolute_error(y_test, y_pred)
    RMSE = np.sqrt(MSE)

    print(residual)
    print('RMSE: ', RMSE)
