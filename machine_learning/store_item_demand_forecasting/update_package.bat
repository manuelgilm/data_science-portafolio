call cd demand_forecasting/
call python -m build
call cd dist/
call pip install demand_forecasting-0.1-py3-none-any.whl --force-reinstall
call cd ../

