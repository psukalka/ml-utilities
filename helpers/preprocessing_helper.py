import numpy as np
import pandas as pd

class PreprocessingHelper:
    @staticmethod
    def import_dataset(file_name='Data.csv'):
        dataset = pd.read_csv(file_name)
        return dataset

    @staticmethod
    def get_x_y(file_name='Data.csv'):
        dataset = PreprocessingHelper.import_dataset(file_name)
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        return X, y
    
    @staticmethod
    def fill_missing_values(X, col_start=1, col_end=3, strategy='mean'):
        from sklearn.impute import SimpleImputer

        imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
        imputer.fit(X[:, col_start:col_end])
        X[:, col_start:col_end] = imputer.transform(X[:, col_start:col_end])
        return X
    
    @staticmethod
    def one_hot_encode(X):
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder

        ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
        updated_X = np.array(ct.fit_transform(X))
        return updated_X