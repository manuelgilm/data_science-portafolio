from factory_pattern_for_ml_comparison.models.classifiers import CustomRFC
from factory_pattern_for_ml_comparison.data.retrieval import get_train_test_data
from factory_pattern_for_ml_comparison.utils.utils import get_or_create_experiment
from factory_pattern_for_ml_comparison.factories.model_factory import ModelFactory
from sklearn.metrics import classification_report
import mlflow

from factory_pattern_for_ml_comparison.models.mlflow_run import CustomRun


def main():
    x_train, x_test, y_train, y_test = get_train_test_data()

    experiment = get_or_create_experiment(name="CustomRuns")

    mf = ModelFactory()
    model_type = "classifiers"

    my_run = CustomRun(experiment_name="CustomRuns", run_name="CustomRun")
    my_run.log(
        metadata={
            "metrics": {"test": 0.1},
            "params": {"param1": "somthing", "param2": "something2"},
            "tags": {"tag1": "val1", "tag2": "val2"},
        },
        include_object=True,
    )

    # with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
    #     run_id = run.info.run_id
    #     for model in mf.get_model_list(model_type):
    #         model_name = model[0]
    #         model_obj = model[1](model_name)

    #         model_obj.fit(x_train, y_train)
    #         predictions = model_obj.predict(x_test)
    #         print(f"Model: {model_name}")
    #         macro_avg = classification_report(y_test, predictions, output_dict=True)[
    #             "macro avg"
    #         ]
    #         prefix = f"{model_name}_test_"
    #         macro_avg_ = {prefix + key: value for key, value in macro_avg.items()}
    #         print("\n")

    #         model_obj.log_to_mlflow(metadata={"metrics": macro_avg_})
