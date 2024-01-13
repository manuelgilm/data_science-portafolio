from movie_review_classification.training.train import  get_predictions
from movie_review_classification.data_preparation.data_preparation import get_data
def main():
    train_data, test_data = get_data()
    predictions = get_predictions(base_model="distilbert-base-uncased", dataset=test_data)
    print(predictions)