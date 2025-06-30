def initial_testing():
    from helpers.preprocessing_helper import PreprocessingHelper

    X, y = PreprocessingHelper.get_x_y(r"C:\Users\Pawan\Downloads\Machine-Learning-A-Z-Codes-Datasets\Machine Learning A-Z\Part 1 - Data Preprocessing\Section 2 -------------------- Part 1 - Data Preprocessing --------------------\Python\Data.csv")
    print(f"--- Initial Values ---")
    print(X)
    print(y)
    print(f"--- After filling missing data ---")
    X = PreprocessingHelper.fill_missing_values(X, col_start=1, col_end=3)
    print(X)
    print(f"--- After One Hot Encoding data ---")
    X = PreprocessingHelper.one_hot_encode(X)
    print(X)

if __name__ == "__main__":
    initial_testing()