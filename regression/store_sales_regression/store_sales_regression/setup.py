from setuptools import find_packages
from setuptools import setup

setup(
    name="store_sales_regression",
    version="0.1",
    packages=[
        "store_sales_regression.data",
        "store_sales_regression.utils",
        "store_sales_regression.regression",
    ],
    description="EDSA regression challenge",
    author_email="manuelgilsitio@gmail.com",
)
