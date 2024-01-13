from movie_review_classification.configs.conf import get_config


def main():
    config = get_config(path="training_config.yaml")
    print(config)