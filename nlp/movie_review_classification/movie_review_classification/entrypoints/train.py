from movie_review_classification.configs.conf import get_config
from movie_review_classification.data_preparation.data_preparation import get_data
from movie_review_classification.data_preparation.data_preparation import tokenize_data
from movie_review_classification.training.train import train_model

from transformers import AutoTokenizer
def main():
    config = get_config(path="training_config.yaml")
    train_data, test_data = get_data()
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    train_data = tokenize_data(train_data, tokenizer)
    test_data = tokenize_data(test_data, tokenizer)
    train_model(
        train_dataset=train_data,
        eval_dataset=test_data,
        configs=config,
        tokenizer=tokenizer
    )