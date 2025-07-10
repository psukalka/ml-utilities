from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR

class RegressionHelper:
    @staticmethod
    def get_linear_regressor():
        regressor = LinearRegression()
        return regressor
    
    @staticmethod
    def train_with_linear_regressor(X_train, y_train):
        trained_regressor = RegressionHelper.get_linear_regressor()
        trained_regressor.fit(X_train, y_train)
        return trained_regressor
    
    @staticmethod
    def predict_with_linear_regressor(trained_regressor, X_test):
        return trained_regressor.predict(X_test)
    
    @staticmethod
    def get_polynomial_features(X, degree=2):
        poly_features = PolynomialFeatures(degree=degree)
        X_poly = poly_features.fit_transform(X)
        return X_poly, poly_features

    @staticmethod
    def train_with_poly_regressor(X_train, y_train, degree=2):
        X_poly, poly_features = RegressionHelper.get_polynomial_features(X_train, degree=degree)
        poly_reg = RegressionHelper.train_with_linear_regressor(X_poly, y_train)
        return poly_reg, poly_features
    
    @staticmethod
    def train_with_svr(scaled_X_train, scaled_y_train, kernel='rbf'):
        regressor = SVR(kernel=kernel)
        regressor.fit(scaled_X_train, scaled_y_train)
        return regressor
