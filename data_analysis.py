import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from pandas.plotting import table


def descriptive_analysis(df):
    # Display summary statistics of the numerical features
    st.subheader("Descriptive Analysis")
    st.write(df.describe())

    desc = df.describe()
    #create a subplot without frame
    plt.figure(figsize=(15,2))
    plot = plt.subplot(frame_on=False)
    #remove axis
    plot.xaxis.set_visible(False) 
    plot.yaxis.set_visible(False) 
    #create the table plot and position it in the upper left corner
    table(plot, desc,loc='upper right')
    #save the plot as a png file
    plt.savefig('download/desc_plot.png')

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
    
    st.subheader("Data Visualization")
    st.pyplot()
    plt.savefig('download/plot.png')