import streamlit as st
from PIL import Image

st.set_page_config(layout="wide")


# Ścieżka do wybranego zdjęcia
image_path = "app/predictions.png"  # Zmień na ścieżkę do swojego zdjęcia


properties = {
    "Water Coverage": "38% of image",  # Przykładowa wartość "Water Coverage"
    "Building Non-Flooded": "27 areas",  # Przykładowa wartość "Building Non-Flooded"
    "Building Flooded": "42 areas",  # Przykładowa wartość "Building Flooded"
    "Road Non-Flooded": "73% length",  # Przykładowa wartość "Road Non-Flooded"
    "Road Flooded": "27% length",  # Przykładowa wartość "Road Flooded"
}

# Załadowanie obrazu
image = Image.open(image_path)

# Podziel stronę na dwie kolumny
st.write("# Natural Distaster Analysis:")
st.write(" ")
left_column, _, right_column = st.columns([2, 1, 5])

# W lewej kolumnie wyświetl informacje o właściwościach obrazu
with left_column:
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")

    st.write("### Image properties:")
    st.dataframe(properties, column_config=None)

    st.write(" ")
    st.write(" ")

    st.write("##### Black line shows the best route to get to the choosen building")


# W prawej kolumnie wyświetl obraz
with right_column:
    st.write(" ")

    st.image(image, width=700)
