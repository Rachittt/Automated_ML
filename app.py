import os
import shutil
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

upload_container = st.container()
descibe_container = st.container()
plot_container = st.empty()
model_container = st.container()
download_container = st.container()

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
    upload_container.dataframe(df)

    # Drop columns
    cols_to_drop = upload_container.multiselect("Select columns to drop:", df.columns)
    df = df.drop(cols_to_drop, axis=1)

    # Select target column
    target_column = upload_container.selectbox("Select the target column:", df.columns)



    if 'model_button' not in st.session_state:
        st.session_state.model_button = False

    def click_model_button():
        st.session_state.model_button = not st.session_state.model_button



    if descibe_container.button("Proceed and Visualize Data", key='visualize', on_click=click_model_button):
      Path("download").mkdir(parents=True, exist_ok=True)
      with st.spinner('Please Wait...'):
        descriptive_analysis(df, descibe_container)
        visualize_data(df, plot_container)



    if 'download_button' not in st.session_state:
        st.session_state.download_button = False

    def click_download_button():
        st.session_state.download_button = not st.session_state.download_button



    if st.session_state.model_button:
        if model_container.button("Build Regression Model", key='reg_model', on_click=click_download_button):
            with st.spinner('Please Wait...'):
                df, X, y = preprocess_data(df, target_column)
                X_train, X_test, y_train, y_test = scale_encode_split(X, y)
                save_preprocessed_data(X_train, X_test, y_train, y_test)
                best_model, best_model_name, best_model_score = tune_regression_model(X_train, y_train)
                model_container.write("Best Model is: " + best_model_name)
                print_regression_scores(best_model, X_test, y_test, model_container)
                final_model(X, y, best_model)

        elif model_container.button("Build Classification Model", key='cla_model', on_click=click_download_button):
            with st.spinner('Please Wait...'):
                df, X, y = preprocess_data(df, target_column)
                X_train, X_test, y_train, y_test = scale_encode_split(X, y)
                save_preprocessed_data(X_train, X_test, y_train, y_test)
                best_model, best_model_name, best_model_score = tune_classification_model(X_train, y_train)
                model_container.write("Best Model is: " + best_model_name)
                print_classification_scores(best_model, X_test, y_test, model_container)
                final_model(X, y, best_model)



    if 'refresh_button' not in st.session_state:
        st.session_state.refresh_button = False

    def click_refresh_button():
        st.session_state.refresh_button = not st.session_state.refresh_button



    if st.session_state.download_button:
        # Checkbox selection
        files_to_download = download_container.multiselect(
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

        with zipfile.ZipFile("download/download.zip", "w") as zipf:
            if "All Files" in files_to_download:
                for filename in all_files:
                    filepath = f"download/{filename}"
                    try:
                        zipf.write(filepath)
                    except FileNotFoundError:
                        download_container.error(f"File not found")
            else:
                for filename in files_to_download:
                    filepath = f"download/{filename}"
                    try:
                        zipf.write(filepath)
                    except FileNotFoundError:
                        download_container.error(f"File not found: {filename}")

        # with open("download/download.zip", "rb") as file:
        #     btn = download_container.download_button(
        #             label="Download ZIP",
        #             data=file,
        #             file_name="download.zip",
        #             mime="application/zip"
        #         )
                        
        file = open("download/download.zip", "rb")
        download_container.download_button(label="Download ZIP", data=file, file_name="download.zip", mime="application/zip", on_click=click_refresh_button)
        file.close()



    if st.session_state.refresh_button:
        if st.button("Build New Model"):
            try:
                shutil.rmtree("download")  # Deletes the download folder and its contents
            except FileNotFoundError:
                st.error(f"File not found")
            uploaded_file = None
            click_refresh_button()
            click_download_button()
            click_model_button()
            upload_container = st.container()
            descibe_container = st.container()
            plot_container = st.empty()
            model_container = st.container()
            download_container = st.container()
            st.rerun()


  except Exception as e:
    upload_container.error(f"Error reading the file: {e}")