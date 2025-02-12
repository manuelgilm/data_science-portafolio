from train_service.src.train import Trainer

def main():
    train_service = Trainer()
    x_train, x_test, y_train, y_test = train_service.get_train_test_data()
    print(x_train.head())
