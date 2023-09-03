# Description: This file is used to run the program.
from training_package.custom_regressor import MultiRegressor
from training_package.training import get_train_test_data

if __name__=="__main__":

    x_train1, x_test1, y_train1, y_test1 = get_train_test_data(n_features=20)
    x_train2, x_test2, y_train2, y_test2 = get_train_test_data(n_features=20)
    X_train = (x_train1, x_train2)
    X_test = (x_test1, x_test2)
    Y_train = (y_train1, y_train2)
    Y_test = (y_test1, y_test2)

    regressor = MultiRegressor(experiment_name="custom_model_serving")
    run_id = regressor.fit(X_train, Y_train)
    regressor.evaluate(X_test, Y_test, run_id)
    
