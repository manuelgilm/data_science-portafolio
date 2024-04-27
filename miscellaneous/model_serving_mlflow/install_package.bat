
call python -m build training_package/
call cd training_package/dist
call pip install training_package-0.1-py3-none-any.whl --force-reinstall
call cd ..
call cd ..

