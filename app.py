import streamlit as st
from app1 import app as app1
from app2 import app as app2

def main():
    st.sidebar.title("Navigation")
    app_selection = st.sidebar.radio("Select an app", ("🎓 Student Performance Prediction", "🍃 Iris Dataset Clustering and Visualization 🍃"))

    if app_selection == "🎓 Student Performance Prediction":
        app1()
    elif app_selection == "🍃 Iris Dataset Clustering and Visualization 🍃":
        app2()

if __name__ == "__main__":
    main()
