call cd store_sales_regression
call python -m build
call cd dist
call pip install store_sales_regression-0.1-py3-none-any.whl --force-reinstall
call cd ../..
