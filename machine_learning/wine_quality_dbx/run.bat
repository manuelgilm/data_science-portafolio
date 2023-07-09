@REM call dbx execute --cluster-id=%1 --job=feature_preparation --no-package
@REM call dbx execute --cluster-id=%1 --job=training --no-package
call dbx execute --cluster-id=%1 --job=inferencing --no-package
