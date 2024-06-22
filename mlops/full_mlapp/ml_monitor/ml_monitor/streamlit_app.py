from ml_monitor.utils.utils import get_root_path
from ml_monitor.utils.utils import get_payload
from ml_monitor.utils.utils import get_record
from ml_monitor.inference.model import get_model_prediction

from streamlit_gsheets import GSheetsConnection

import streamlit as st 
import pandas as pd

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

label = st.sidebar.selectbox("Species", ["Setosa", "Versicolor", "Virginica", "Unknown"])
conn = st.connection("gsheets", type=GSheetsConnection)
WORKSHEET = "Hoja 1"

if st.sidebar.button("Predict"):
    st.write("Predicting...")
    # print the payload
    payload = get_payload(sepal_length, sepal_width, petal_length, petal_width)
    prediction = get_model_prediction(payload)
    st.write(f"Flower Species: {species_map[prediction['predictions'][0]]}")
    record = get_record(payload)
    record["prediction"] = species_map[prediction["predictions"][0]]
    record["label"] = label
    # Fetch the data from the Google Sheet
    existing_data = conn.read(worksheet=WORKSHEET, usecols = list(range(6)), ttl=0)
    existing_data = existing_data.dropna(how='all') 

    df_record = pd.DataFrame([record])
    updated_df = pd.concat([existing_data, df_record], ignore_index=True)

    # Update Google Sheets with the new vendor data
    conn.update(worksheet=WORKSHEET, data=updated_df)

    st.success("Prediction completed and saved to Google Sheets!")
