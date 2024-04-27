@REM call dbx execute --cluster-id=%1 --job=feature_preparation --no-package
@REM call dbx execute --cluster-id=%1 --job=training --no-package
@REM call dbx execute --cluster-id=%1 --job=training_crossvalidation --no-package
@REM call dbx execute --cluster-id=%1 --job=training_sklearn --no-package
@REM call dbx execute --cluster-id=%1 --job=training_sklearn_crossvalidation --no-package
@REM call dbx execute --cluster-id=%1 --job=inferencing --no-package
call dbx execute --cluster-id=%1 --job=inference_sklearn --no-package
@REM call dbx execute --cluster-id=%1 --job=transition_model --no-package
@REM call dbx execute --cluster-id=%1 --job=delete_all --no-package

