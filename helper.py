import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


def max_min_normalize(X):
    return (X - X.min()) / (X.max() - X.min())


def mean_normalize(X):
    return (X - X.mean()) / X.std()


def read_data(path: str, label: str, dropped_columns):
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