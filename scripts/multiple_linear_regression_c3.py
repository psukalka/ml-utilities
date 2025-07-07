from helpers.preprocessing_helper import PreprocessingHelper as ph
from helpers.regression_helper import RegressionHelper as rh

def multiple_linear_regression():
    X, y = ph.get_x_y(r"C:\Users\Pawan\Downloads\Machine-Learning-A-Z-Codes-Datasets\Machine Learning A-Z\Part 2 - Regression\Section 5 - Multiple Linear Regression\Python\50_Startups.csv")
    print(f"--- Initial Values ---")
    print(X)
    print(y)
    print(f"--- One Hot Encoding on States Value ---")
    X = ph.one_hot_encode(X, column=3)
    print(X)
    print(f"--- Split data ---")
    X_train, X_test, y_train, y_test = ph.split_data(X, y)
    print(f"--- Training regression model ---")
    trained_regressor = rh.train_with_linear_regressor(X_train=X_train, y_train=y_train)
    print(f"--- Predicting Results ---")
    y_pred = trained_regressor.predict(X_test)
    print(f"--- Compare results ---")
    ph.compare_results_vertically(y_pred, y_test)


if __name__ == "__main__":
    multiple_linear_regression()