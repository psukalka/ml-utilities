from sklearn.linear_model import LinearRegression

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
