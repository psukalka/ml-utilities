import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from helpers.preprocessing_helper import PreprocessingHelper as ph
from helpers.regression_helper import RegressionHelper as rh


def decision_tree_regression():
    print(f"--- Reading data ---")
    X, y = ph.get_x_y(r"C:\Users\Pawan\Downloads\Machine-Learning-A-Z-Codes-Datasets\Machine Learning A-Z\Part 2 - Regression\Section 7 - Support Vector Regression (SVR)\Python\Position_Salaries.csv", feature_range=slice(1, -1))
    print(X)
    print(y)
    print(f"--- Train with Decision Tree Regressor ---")
    reg = rh.train_with_dtr(X, y)
    print(f"--- Predicting with DTR ---")
    print(reg.predict([[6.5]]))
    print(f"--- Visualizing DTR ---")
    visualize_dtr(X, y, reg)

def visualize_dtr(X, y, reg):
    X_grid = np.arange(min(X), max(X), 0.1)
    X_grid = X_grid.reshape(len(X_grid), 1)
    plt.scatter(X, y, color='red')
    plt.plot(X_grid, reg.predict(X_grid), color='blue')
    plt.title('Truth or Bluff (DTR)')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.savefig('dtr_pred_c6.png')
    plt.close()
    

if __name__ == "__main__":
    decision_tree_regression()