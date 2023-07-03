@REM call dbx execute --cluster-id=%1 --job=inference_pipeline --no-package
@REM call dbx execute --cluster-id=%1 --job=delete --no-package
@REM call dbx execute --cluster-id=%1 --job=feature_preparation --no-package
call dbx execute --cluster-id=%1 --job=pyspark_pipeline --no-package
call dbx execute --cluster-id=%1 --job=crossvalidation_pipeline --no-package