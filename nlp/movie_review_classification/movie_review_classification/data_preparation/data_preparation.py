from datasets import load_dataset
from datasets import Dataset
from typing import Tuple


def get_data(
    training_ratio: int = 1, testing_ratio: int = 1
) -> Tuple[Dataset, Dataset]:
    """
    Get the data from the imdb dataset.

    :param training_ratio: Ratio of the training data.
    :param testing_ratio: Ratio of the testing data.
    :return: Train and test data.
    """
    train_data = load_dataset("imdb", split=f"train[:{training_ratio}%]")
    test_data = load_dataset("imdb", split=f"test[:{testing_ratio}%]")
    return train_data, test_data

def tokenize_data(data:Dataset, tokenizer):
    """
    Tokenize data.
    :param data: Dataset.
    :param tokenizer: Tokenizer.
    :return: Tokenized data.
    """

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    tokenized_data = data.map(tokenize_function, batched=True)

    return tokenized_data