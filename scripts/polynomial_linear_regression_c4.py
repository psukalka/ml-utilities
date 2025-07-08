import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from helpers.preprocessing_helper import PreprocessingHelper as ph
from helpers.regression_helper import RegressionHelper as rh

def polynomial_linear_regression():
    print(f"--- Reading data ---")
    X, y = ph.get_x_y(r"C:\Users\Pawan\Downloads\Machine-Learning-A-Z-Codes-Datasets\Machine Learning A-Z\Part 2 - Regression\Section 6 - Polynomial Regression\Python\Position_Salaries.csv", feature_range=slice(1,-1))
    print(X)
    print(y)
    print(f"--- Training LR model ---")
    lin_reg = rh.train_with_linear_regressor(X, y)
    print(f"--- Train with Polynomial regression ---")
    poly_reg, poly_features = rh.train_with_poly_regressor(X, y, degree=4)
    print(f"--- Visualizing Linear Regression ---")
    visualize_lin_reg(X, y, lin_reg)
    print(f"--- Visualizing Polynomial Regression ---")
    visualize_poly_reg(X, y, poly_reg)
    print(f"--- Predicting with Linear and Polynomial regression ---")
    print(f"Linear prediction: {lin_reg.predict([[6.5]])}")
    print(f"Polynomial prediction: {poly_reg.predict(poly_features.transform([[6.5]]))}")


def visualize_lin_reg(X, y, lin_reg):
    plt.scatter(X, y, color='red')  # Print dots
    plt.plot(X, lin_reg.predict(X), color='blue')  # Print a linear line 
    plt.title('Truth or Bluff (Linear Regression)')
    plt.xlabel('Position Level')
    plt.ylabel('Salary')
    plt.savefig('lin_reg_pred_c4.png')
    plt.close()

def visualize_poly_reg(X, y, poly_reg):
    plt.scatter(X, y, color='red')
    X_poly, poly_features = rh.get_polynomial_features(X, degree=4)
    plt.plot(X, poly_reg.predict(X_poly), color='blue')
    plt.title('Truth or Bluff (Polynomial Regression)')
    plt.xlabel('Position Level')
    plt.ylabel('Salary')
    plt.savefig('poly_reg_pred_c4.png')
    plt.close()


if __name__ == "__main__":
    polynomial_linear_regression()