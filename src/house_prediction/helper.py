import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


def max_min_normalize(X):
    return (X - X.min()) / (X.max() - X.min())


def mean_normalize(X):
    return (X - X.mean()) / X.std()


def read_data(path):
    df = pd.read_csv(path)
    return df


def drop_columns(df, columns):
    return df.drop(columns, axis=1)


def split_test_train(split_rate: int, X, y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=split_rate / 100, random_state=1, shuffle=False)
    return x_train, x_test, y_train, y_test


def polynomial_transform_train_data(polynom_degree: int, data):
    poly = PolynomialFeatures(degree=polynom_degree)
    x_data_poly = poly.fit_transform(data)
    return x_data_poly


def remove_outlier_with_boxplot(df):
    Q1 = df['Y house price of unit area'].quantile(0.25)
    Q3 = df['Y house price of unit area'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return df[~((df['Y house price of unit area'] < lower_bound) | (df['Y house price of unit area'] > upper_bound))]
