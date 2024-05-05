from factory_pattern_for_ml_comparison.models.classifiers import CustomRFC
from factory_pattern_for_ml_comparison.data.retrieval import get_train_test_data
from factory_pattern_for_ml_comparison.utils.utils import get_or_create_experiment
from factory_pattern_for_ml_comparison.models.model_factory import ModelFactory


def main():

    # experiment = get_or_create_experiment(name="RandomForestClassifier")
    mf = ModelFactory()
    model_type = "classifiers"
    print(mf.get_model_list(model_type=model_type))
    # x_train, x_test, y_train, y_test = get_train_test_data()
    # print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    # model = CustomRFC(model_name="RandomForest")
    # model.fit(x_train, y_train)
    # y_pred = model.predict(x_test)

    # print(classification_report(y_test, y_pred))
