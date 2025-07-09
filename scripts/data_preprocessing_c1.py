def initial_testing():
    from helpers.preprocessing_helper import PreprocessingHelper as ph

    X, y = ph.get_x_y(r"C:\Users\Pawan\Downloads\Machine-Learning-A-Z-Codes-Datasets\Machine Learning A-Z\Part 1 - Data Preprocessing\Section 2 -------------------- Part 1 - Data Preprocessing --------------------\Python\Data.csv")
    print(f"--- Initial Values ---")
    print(X)
    print(y)
    print(f"--- X - After filling missing data ---")
    X = ph.fill_missing_values(X, col_start=1, col_end=3)
    print(X)
    print(f"--- X - After One Hot Encoding data ---")
    X = ph.one_hot_encode(X)
    print(X)
    print(f"--- y - After label encoding ---")
    y = ph.label_encode(y)
    print(y)
    print(f"--- X, y - After splitting data ---")
    X_train, X_test, y_train, y_test = ph.split_data(X, y)
    print(len(X), len(y), len(X_train), len(X_test), len(y_train), len(y_test))
    print(f"--- After feature scaling ---")
    X_train, sc_X = ph.scale_feature(X_train, col_start=3)
    X_test, _ = ph.scale_feature(X_test, sc=sc_X, col_start=3)
    print(X_train)
    print('-'*20)
    print(X_test)

if __name__ == "__main__":
    initial_testing()