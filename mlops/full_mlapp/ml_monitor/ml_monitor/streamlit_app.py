from ml_monitor.utils.utils import get_root_path
from ml_monitor.utils.utils import get_payload
from ml_monitor.inference.model import get_model_prediction
import streamlit as st 

species_map = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
}
st.markdown("# Iris Classifier")
st.markdown("This is a simple Iris classifier that uses a Random Forest model to predict the species of an Iris flower based on its sepal and petal measurements.")
# add local image
image_path = get_root_path() / "ml_monitor" / "images" / "iris-dataset.png"
st.image(image_path.as_posix(), width=800, caption="Iris dataset")

st.sidebar.markdown("## Input Parameters")
st.sidebar.markdown("Please specify the different parameters of the Iris flower to get the prediction from the model.")

sepal_length = st.sidebar.slider("Sepal Length (cm)", 0.0, 10.0, 5.0)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 0.0, 10.0, 5.0)
petal_length = st.sidebar.slider("Petal Length (cm)", 0.0, 10.0, 5.0)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.0, 10.0, 5.0)

if st.sidebar.button("Predict"):
    st.write("Predicting...")
    # print the payload
    payload = get_payload(sepal_length, sepal_width, petal_length, petal_width)
    prediction = get_model_prediction(payload)
    st.write(f"Flower Species: {species_map[prediction['predictions'][0]]}")