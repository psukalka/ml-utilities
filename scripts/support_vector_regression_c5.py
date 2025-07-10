from helpers.preprocessing_helper import PreprocessingHelper as ph
from helpers.regression_helper import RegressionHelper as rh

def support_vector_regression():
    print(f"--- Reading data ---")
    X, y = ph.get_x_y(r"C:\Users\Pawan\Downloads\Machine-Learning-A-Z-Codes-Datasets\Machine Learning A-Z\Part 2 - Regression\Section 7 - Support Vector Regression (SVR)\Python\Position_Salaries.csv", feature_range=slice(1, -1))
    print(X)
    print(y)
    print(f"--- Transformed y ---")
    # Feature scaling expects a 2D array so transform y first
    y = ph.horizontal_to_vertical(y)
    print(y)
    print(f"--- Scaled features ---")
    X, sc_X = ph.scale_feature(X, col_start=0)
    y, sc_Y = ph.scale_feature(y, col_start=0)
    print(X)
    print(y)
    print(f"--- Training with SVR ---")
    svr_reg = rh.train_with_svr(X, y)
    print(f"--- Predicting with SVR ---")
    position = 6.5
    trans_position, _ = ph.scale_feature([[position]], sc=sc_X)
    trans_salary = svr_reg.predict(trans_position)
    salary = sc_Y.inverse_transform(trans_salary.reshape(-1, 1))
    print(f"For position: {position}, predicted salary is: {salary}")


if __name__ == "__main__":
    support_vector_regression()