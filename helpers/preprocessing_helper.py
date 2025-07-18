import numpy as np
import pandas as pd

class PreprocessingHelper:
    @staticmethod
    def import_dataset(file_name='Data.csv'):
        dataset = pd.read_csv(file_name)
        return dataset

    @staticmethod
    def get_x_y(file_name='Data.csv', feature_range=slice(None, -1)):
        dataset = PreprocessingHelper.import_dataset(file_name)
        X = dataset.iloc[:, feature_range].values
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
    def one_hot_encode(X, column=0):
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder

        ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [column])], remainder='passthrough')
        updated_X = np.array(ct.fit_transform(X))
        return updated_X
    
    @staticmethod
    def label_encode(y):
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        return le.fit_transform(y)
    
    @staticmethod
    def split_data(X, y, test_size=0.2):
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def scale_feature(R, sc=None, col_start=None):
        from sklearn.preprocessing import StandardScaler

        if sc is None:
            sc = StandardScaler()
            if col_start:
                R[:, col_start:] = sc.fit_transform(R[:, col_start:])
            else:
                R = sc.fit_transform(R)
        else:
            if col_start:
                R[:, col_start:] = sc.transform(R[:, col_start:])
            else:
                R = sc.transform(R)
        return R, sc

    @staticmethod
    def compare_results_vertically(y_pred, y_test):
        np.set_printoptions(precision=2)
        print(np.concatenate((y_pred.reshape(len(y_pred), 1),
                              y_test.reshape(len(y_test), 1)),
                              axis=1))
        
    @staticmethod
    def horizontal_to_vertical(y):
        return y.reshape(len(y), 1)