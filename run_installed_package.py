from house_prediction import models

print('--------Train-------')
models.train_regression_model(data_path='./RealEstate.csv', split_rate=20, label_name='Y house price of unit area',
                              model_name='house_prediction',
                              dropped_columns=['No', 'X1 transaction date', 'X2 house age'])

print('--------Test--------')
models.predict(data_path='./RealEstate.csv', label_name='Y house price of unit area', model_name='house_prediction',
               dropped_columns=['No', 'X1 transaction date', 'X2 house age'])
