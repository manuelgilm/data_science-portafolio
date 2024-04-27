call databricks fs mkdirs dbfs:/FileStore/configs
call dbfs cp configs/feature_preparation.yaml dbfs:/FileStore/configs/feature_preparation.yaml --overwrite

call python -m build custom_packages/utils/

call databricks fs mkdirs dbfs:/FileStore/wheels 
call databricks fs cp custom_packages/utils/dist/utils-0.0.1-py3-none-any.whl dbfs:/FileStore/wheels/utils-0.0.1-py3-none-any.whl --overwrite
call databricks libraries install --cluster-id %1 --whl dbfs:/FileStore/wheels/utils-0.0.1-py3-none-any.whl