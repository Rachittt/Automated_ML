import os
import streamlit as st
import pandas as pd
from data_analysis import *
from data_preprocessing import *
from hyperparameter_tuning import *
import zipfile
from pathlib import Path


st.set_option('deprecation.showPyplotGlobalUse', False)

# Title and instructions
st.title("Machine Learning App")
st.write("Choose a dataset in CSV, Excel, JSON, or other common formats.")

# File uploader
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "json", "txt"])

if uploaded_file is not None:
  try:
    # Read the data based on file extension
    if uploaded_file.name.endswith(".csv"):
      df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
      df = pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith(".json"):
      df = pd.read_json(uploaded_file)
    else:
      df = pd.read_table(uploaded_file)

    # Display the data
    st.dataframe(df)

    # Drop columns
    cols_to_drop = st.multiselect("Select columns to drop:", df.columns)
    df = df.drop(cols_to_drop, axis=1)

    # Select target column
    target_column = st.selectbox("Select the target column:", df.columns)

    if st.button("Proceed and Visualize Data"):
      Path("download").mkdir(parents=True, exist_ok=True)
      st.session_state.df = df  # Store the updated dataframe
      st.session_state.target_column = target_column  # Store the target column
      st.session_state.page = "model_options"
      with st.spinner('Please Wait...'):
        descriptive_analysis(df)
        visualize_data(df)

  except Exception as e:
    st.error(f"Error reading the file: {e}")

# New page for model building options
if "page" in st.session_state and st.session_state.page == "model_options":
  st.title("Choose an Action")

  df = st.session_state.df
  target_column = st.session_state.target_column
  
  if st.button("Build Regression Model", key='reg_model'):
    with st.spinner('Please Wait...'):
      df, X, y = preprocess_data(df, target_column)
      X_train, X_test, y_train, y_test = scale_encode_split(X, y)
      save_preprocessed_data(X_train, X_test, y_train, y_test)
      best_model, best_model_name, best_model_score = tune_regression_model(X_train, y_train)
      st.write("Best Model is: " + best_model_name)
      print_regression_scores(best_model, X_test, y_test)
      final_model(X, y, best_model)
      st.session_state.page = "download_options"

  elif st.button("Build Classification Model", key='cla_model'):
    with st.spinner('Please Wait...'):
      df, X, y = preprocess_data(df, target_column)
      X_train, X_test, y_train, y_test = scale_encode_split(X, y)
      save_preprocessed_data(X_train, X_test, y_train, y_test)
      best_model, best_model_name, best_model_score = tune_classification_model(X_train, y_train)
      print_classification_scores(best_model, X_test, y_test)
      final_model(X, y, best_model)
      st.session_state.page = "download_options"


# New page for download options
if "page" in st.session_state and st.session_state.page == "download_options":
  # Checkbox selection
  files_to_download = st.multiselect(
    "Select files to download:",
    [
      "All Files",
      "preprocessed_data.csv",
      "X_train.csv",
      "X_test.csv",
      "y_train.csv",
      "y_test.csv",
      "describe.png",
      "plots.png",
      "model.joblib",
    ]
  )

  all_files = [
    "preprocessed_data.csv",
    "X_train.csv",
    "X_test.csv",
    "y_train.csv",
    "y_test.csv",
    "describe.png",
    "plots.png",
    "model.joblib",
  ]

  # if st.download_button("Download Selected Files"):
  with zipfile.ZipFile("download/download.zip", "w") as zipf:
    if "All Files" in files_to_download:
      for filename in all_files:
        filepath = f"download/{filename}"
        try:
          zipf.write(filepath)
        except FileNotFoundError:
          st.error(f"File not found")
    else:
      for filename in files_to_download:
        filepath = f"download/{filename}"
        try:
          zipf.write(filepath)
        except FileNotFoundError:
          st.error(f"File not found: {filename}")

  # st.download_button("Download ZIP", "download/download.zip")

  with open("download/download.zip", "rb") as file:
    btn = st.download_button(
            label="Download ZIP",
            data=file,
            file_name="download.zip",
            mime="application/zip"
          )
