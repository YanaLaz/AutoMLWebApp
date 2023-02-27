#Core packages
import streamlit as st

#EDA packages (Exploratory data analysis)
import pandas as pd
import os
import numpy as np

# Data vizualisation packges
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

# import profiling capability
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report

# ML
from pycaret.classification import setup, compare_models, pull, save_model
from sklearn import model_selection


def main():
    """AutoMl App"""
    # st.title("AutoMl App")


    with st.sidebar:
        st.image("https://www.clearrisk.com/hubfs/data%20analytics%20challenges.jpg")
        st.title("AutoStreamML")
        choice = st.radio("Navigation", ["Upload", "Profiling", "Plot", "ML", "Download"])
        st.info("This application allows you to build an automated ML pipeline using Streamlit, Pandas profiling and PyCaret!")

    if os.path.exists("sourcedata.csv"):
        df = pd.read_csv('sourcedata.csv', index_col=None)



    if choice == "Upload":
        st.title('Upload your data for modelling',)
        file = st.file_uploader("Upload your dataset", type=['csv','txt','xls'])
        if file:
            df = pd.read_csv(file, index_col=None)
            df.to_csv("sourcedata.csv", index=None)
            st.dataframe(df)

    if choice == "Profiling":
        st.title('Automated exploring data analysis')
        profile_report = df.profile_report()
        st_profile_report(profile_report)

    if choice == "Plot":
        st.title('Data Visualization')
        if st.checkbox('Correlation matrix with Seaborn'):
            st.write(sns.heatmap(df.corr(), annot=True))
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()

        if st.checkbox('Pie Chart'):
            all_columns = df.columns.to_list()
            column_to_plot = st.selectbox('Select 1 column', all_columns)
            pie_plot = df[column_to_plot].value_counts().plot.pie(autopct="%1.1f%%")
            st.write(pie_plot)
            st.pyplot()

        all_columns = df.columns.to_list()
        type_of_plot = st.selectbox('Select type of the plot', ["area","bar",'line','hist','box','kde'])
        selected_columns_names = st.multiselect('Select columns to plot', all_columns)

        if st.button('Generate plot'):
            st.success('Generating customizable plot of {} for {}'.format(type_of_plot,selected_columns_names))

            # Plot by Streamlit
            if type_of_plot == "area":
                cust_data = df[selected_columns_names]
                st.area_chart(cust_data)
            elif type_of_plot == "bar":
                cust_data = df[selected_columns_names]
                st.bar_chart(cust_data)
            elif type_of_plot == "line":
                cust_data = df[selected_columns_names]
                st.line_chart(cust_data)

            # Custom plot
            elif type_of_plot:
                cust_plot = df[selected_columns_names].plot(kind=type_of_plot)
                st.write(cust_plot)
                st.pyplot()

    if choice == "ML":
        st.title('Builing ML Model')
        target = st.selectbox('Select your target', df.columns)
        st.title('Builing ML Model')
        model = st.selectbox('Select your model', ['Classification', 'Regression', 'Clustering'])
        if st.button("Train model"):
            setup(df, target = target, silent = True)
            setup_df = pull()
            st.info("This is the ML settings")
            st.dataframe(setup_df)
            best_model = compare_models()
            compare_df = pull()
            st.info('This is the ML Model:')
            st.dataframe(compare_df)
            best_model
            save_model(best_model, 'best_model')

        # Model building
        X = df.iloc[:,0:-1]
        Y = df.iloc[:,-1]
        seed = 7

        # Model
        models = []



    if choice == "Download":
        with open ('best_model.pkl', 'rb') as f:
            st.download_button('Download the Model', f, "trained_model.pkl")


if __name__ == '__main__':
    main()




