call databricks fs mkdirs dbfs:/FileStore/configs/wine_quality
call dbfs cp configs/common_params.yaml dbfs:/FileStore/configs/wine_quality/common_params.yaml --overwrite

call databricks fs cp -r local_data/wine_quality dbfs:/FileStore/datasets/wine_quality --overwrite
call python -m build wine_quality/
call databricks fs cp wine_quality/dist/wine_quality-0.1-py3-none-any.whl dbfs:/FileStore/wheels/wine_quality/wine_quality-0.1-py3-none-any.whl --overwrite
call databricks libraries install --cluster-id %1 --whl dbfs:/FileStore/wheels/wine_quality/wine_quality-0.1-py3-none-any.whl