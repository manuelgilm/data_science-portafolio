import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from iris_classifier.data.retrieval import get_dataset
from iris_classifier.utils.utils import get_project_root
from sklearn import datasets

# Assuming you have loaded iris dataset
df = get_dataset()

# Add your image path
root_path = get_project_root()
header_image_path = (
    root_path / "iris_classifier/eda/iris-dataset.png"
)  # Change "your_image_path.png" to the actual path of your image
header_image = plt.imread(header_image_path)

# Title of your web app with an image in the header
st.title("Exploratory Data Analysis with Iris Dataset")
st.image(header_image, use_column_width=True)

# Display the dataframe
st.subheader("Dataframe")
st.write(df)

# Summary statistics
st.subheader("Summary Statistics")
st.write(df.describe())

# Correlation heatmap
st.subheader("Correlation Heatmap")
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
correlation_heatmap = plt.gcf()  # Get the current figure explicitly
st.pyplot(correlation_heatmap)

# Scatter plot
st.subheader("The Iris dataset")
iris = datasets.load_iris()
_, ax = plt.subplots()
scatter = ax.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target)
ax.set(xlabel=iris.feature_names[0], ylabel=iris.feature_names[1])
_ = ax.legend(
    scatter.legend_elements()[0],
    iris.target_names,
    loc="lower right",
    title="Classes",
)
scatter_plot = plt.gcf()
st.pyplot(scatter_plot)

# Pairplot
st.subheader("Pairplot")
pairplot = sns.pairplot(df, hue="target")
pairplot_figure = pairplot.fig  # Get the figure from the pairplot explicitly
st.pyplot(pairplot_figure)

# Bar plot of target variable distribution
st.subheader("Target Variable Distribution")
plt.figure(figsize=(8, 4))
sns.countplot(x="target", data=df)
plt.xlabel("Target Class")
plt.ylabel("Count")
plt.title("Distribution of Target Variable")
target_distribution_plot = plt.gcf()  # Get the current figure explicitly
st.pyplot(target_distribution_plot)
