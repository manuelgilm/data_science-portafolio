call dbfs mkdirs dbfs:/FileStore/configs
call dbfs rm dbfs:/FileStore/configs/feature_preparation.yaml
call dbfs cp configs/feature_preparation.yaml dbfs:/FileStore/configs/feature_preparation.yaml