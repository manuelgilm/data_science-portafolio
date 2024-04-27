from package.configs.utils import get_configs
from package.features.create_features import create_feature_table
from package.features.data_processing import get_feature_dataframe

if __name__ == "__main__":
    configs = get_configs()
    print(configs)
    table_name = configs["table_name"]
    database_name = configs["database_name"]
    sdf = get_feature_dataframe()
    create_feature_table(
        table_name=table_name, database_name=database_name, sdf=sdf
    )
