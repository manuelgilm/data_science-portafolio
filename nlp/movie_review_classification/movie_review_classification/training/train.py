from movie_review_classification.training.mlflow_utils import get_or_create_experiment
from movie_review_classification.configs.conf import get_config
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
from transformers import pipeline
from transformers import DataCollatorWithPadding
from datasets import Dataset
import tempfile
import evaluate

from typing import Dict
from typing import Any
import pkgutil
import yaml
import numpy as np
import mlflow

def compute_metrics(eval_preds):
    """
    Compute metrics.
    """
    metric = evaluate.load("accuracy")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def get_trainer_object(
    base_model: str,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    training_args: Dict[str, Any],
    compute_metrics: callable,
    tokenizer: AutoTokenizer
):
    """
    Get trainer object.

    :param base_model: Base model.
    :param training_args: Training arguments.
    :param train_dataset: Training dataset.
    :param eval_dataset: Evaluation dataset.
    :param compute_metrics: Compute metrics.
    :return: Trainer object.
    """
    # dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    training_args = TrainingArguments(**training_args)
    model = AutoModelForSequenceClassification.from_pretrained(base_model)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    return trainer, tokenizer


def create_pipeline(tokenizer, model_config_path: str, task: str):
    """
    Create pipeline.

    :param trainer: Trainer object.
    :param base_model: Base model.
    :param task: Task.
    :return: Pipeline.
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_config_path)
    pipe = pipeline(task=task, model=model, tokenizer=tokenizer)
    return pipe


def train_model(
    tokenizer: AutoTokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    configs: Dict[str, Any],
):
    """
    Train model. Function to be called in the entrypoint.

    :param base_model: Base model.
    :param training_args: Training arguments.
    :param train_dataset: Training dataset.
    :param eval_dataset: Evaluation dataset.
    :param compute_metrics: Compute metrics.
    :return: Trainer object.
    """
    base_model = configs["base_model"]
    trainer, tokenizer = get_trainer_object(
        base_model=base_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        training_args=configs["training_configs"],
        compute_metrics=compute_metrics,
        tokenizer = tokenizer
    )
    task = "text-classification"
    experiment_id = get_or_create_experiment(
        experiment_name=configs["mlflow_experiment"]["name"]
    )
    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.transformers.autolog()
        trainer.train()
        with tempfile.TemporaryDirectory() as tmpdirname:
            trainer.save_model(tmpdirname)
            pipeline = create_pipeline(
                tokenizer=tokenizer,
                model_config_path=tmpdirname,
                task=task,
            )
            mlflow.log_artifacts(tmpdirname, "model_config")
            mlflow.transformers.log_model(
                transformers_model=pipeline, artifact_path="pipeline", task=task
            )

    return trainer


# def get_pipeline(base_model: str, task: str):
#     """
#     Get pipeline.

#     :return: Pipeline.
#     """
#     tokenizer = AutoTokenizer.from_pretrained(base_model)
#     pipe = pipeline(
#         task=task, model=base_model, tokenizer=base_model, truncation=True, padding=True
#     )

#     return pipe


# def get_predictions(base_model: str, dataset: Dataset):
#     """
#     Get predictions from model.

#     :param base_model: Base model.
#     :param dataset: Dataset.
#     :return: Predictions.
#     """

#     pipe = get_pipeline(base_model=base_model, task="text-classification")
#     predictions = pipe(dataset["text"], batch_size=8)
#     return predictions
