#imports 
import numpy as np

#for machine learning tasks
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

#to save the model
import pickle

#Load the dataset
iris_x, iris_y = datasets.load_iris(return_X_y=True)
print("This dataset has {len(np.unique(iris_y))} labels which are Setosa, Versicolour and Virginica")

#Splitting the dataset
print("Splitting the data...")
x_train, x_test, y_train, y_test = train_test_split(iris_x, iris_y, test_size = 0.2, random_state = 42)

#buildint a simnple MLP classifier
print("Creating the model ...")
mlp_classifier = MLPClassifier(random_state = 1, max_iter = 300).fit(x_train,y_train)

#testing the model
print("testing the model...")
y_pred = mlp_classifier.predict(x_test)

acc = 100*sum([1 if y_p == y_t else 0 for y_p,y_t in zip(y_pred,y_test)])/len(y_test)

print(f"Accuracy is: {acc} %")

#Saving the model
print("Saving the model...")
with open("iris_classifier.pkl","wb") as f:
    pickle.dump(mlp_classifier,f)
print("Model saved!")