import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


def descriptive_analysis(df):
    # Display summary statistics of the numerical features
    # print(df.describe())
    st.subheader("Descriptive Analysis")
    st.write(df.describe())


def visualize_data(df):
    # Set up subplots for numerical features
    num_rows = len(df.columns) // 2 + len(df.columns) % 2
    num_cols = 2
    plt.figure(figsize=(25, 5 * num_rows))

    for i, column in enumerate(df.columns, 1):
        plt.subplot(num_rows, num_cols, i)
        sns.histplot(df[column], kde=True)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')

    # plt.tight_layout()
    # plt.show()
    st.subheader("Data Visualization")
    st.pyplot()
