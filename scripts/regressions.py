import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from helpers.preprocessing_helper import PreprocessingHelper as ph
from helpers.regression_helper import RegressionHelper as rh

def linear_regression():
    X, y = ph.get_x_y(r"C:\Users\Pawan\Downloads\Machine-Learning-A-Z-Codes-Datasets\Machine Learning A-Z\Part 2 - Regression\Section 4 - Simple Linear Regression\Python\Salary_Data.csv")
    print(f"--- Original Data ----")
    print(X)
    print(y)
    print(f"--- Split data ---")
    X_train, X_test, y_train, y_test = ph.split_data(X, y)
    print(len(X), len(y), len(X_train), len(X_test), len(y_train), len(y_test))
    print(f"--- Linear Regression training ---")
    trained_regressor = rh.train_with_linear_regressor(X_train, y_train)
    visualize(X_train, y_train, trained_regressor)
    visualize(X_test, y_test, trained_regressor, type='test')


def visualize(X_train, y_train, trained_regressor, type='training'):
    plt.scatter(X_train, y_train, color='red')
    plt.plot(X_train, rh.predict_with_linear_regressor(trained_regressor, X_train), color='blue')
    plt.title(f'Salary vs Experience ({type} Set)')
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    plt.savefig(f'linear_regression_{type}_data.png')
    plt.close()

if __name__ == "__main__":
    linear_regression()