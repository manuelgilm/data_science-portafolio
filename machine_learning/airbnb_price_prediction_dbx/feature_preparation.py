
data_path = "dbfs:/FileStore/datasets/airbnb/listings.csv"
sdf = spark.read.csv(data_path, header=True, inferSchema=True, multiLine="true", escape = '"')

print(sdf.show())