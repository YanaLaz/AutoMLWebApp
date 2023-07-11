# Core packages
import pandas_profiling
import streamlit as st
import streamlit_authenticator as stauth

# EDA packages (Exploratory data analysis)
import pandas as pd
import os

# Data vizualisation packges
import matplotlib

matplotlib.use('Agg')
import seaborn as sns

# import profiling capability
from streamlit_pandas_profiling import st_profile_report

# ML
from pycaret.classification import setup as setup_cl, compare_models as compare_models_cl, pull as pull_cl, \
    save_model as save_model_cl
from pycaret.regression import setup as setup_rg, compare_models as compare_models_rg, pull as pull_rg, \
    save_model as save_model_rg
from pycaret.clustering import setup as setup_clust, create_model, pull as pull_clust, save_model as save_model_clust, \
    assign_model

import joblib

# Database
import database as db


def main():
    """AutoMl App"""
    # st.title("AutoMl App")

    with st.sidebar:
        st.title(f"Welcome, {name}!")

        st.image("https://www.clearrisk.com/hubfs/data%20analytics%20challenges.jpg")
        st.title("AutoStreamML")
        choice = st.radio("Navigation", ["Upload", "Profiling", "Plot", "ML", "Download", "Prediction"])
        st.info(
            "This application allows you to build an automated ML pipeline!")

        authentificator.logout("Logout", 'sidebar')

    if os.path.exists("sourcedata.csv"):
        df = pd.read_csv('sourcedata.csv', index_col=None)


    if choice == "Upload":
        st.title('Upload your data for modelling')
        file = st.file_uploader("Upload your dataset", type=['csv', 'txt', 'xls', 'xlsx'])
        if file:
            df = pd.read_csv(file, index_col=None)
            df.to_csv("sourcedata.csv", index=None)
            st.dataframe(df)

            filename = file.name

            db.upload_file(username, df, filename)

            if os.path.exists("best_model.pkl"):
                os.remove("best_model.pkl")
            if os.path.exists("best_model_kmeans.pkl"):
                os.remove("best_model_kmeans.pkl")


    if choice == "Profiling":
        st.title('Automated exploring data analysis')
        # profile_report = df.profile_report()
        profile_report = pandas_profiling.ProfileReport(df)
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
        type_of_plot = st.selectbox('Select type of the plot', ["area", "bar", 'line', 'hist', 'box', 'kde'])
        selected_columns_names = st.multiselect('Select columns to plot', all_columns)

        if st.button('Generate plot'):
            st.success('Generating customizable plot of {} for {}'.format(type_of_plot, selected_columns_names))

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
        st.title('Building ML Model')
        target = st.selectbox('Select your target', df.columns)
        model = st.selectbox('Select your model', ['Classification', 'Regression', 'Clustering'])
        train_size = st.slider('Select the percentage of the dataset to use for training', 0, 100, 80)
        if st.button("Train model"):
            if model == 'Classification':
                setup_cl(df, target=target, train_size=train_size/100, silent=True)
                setup_df = pull_cl()
                st.info("This is the ML settings")
                st.dataframe(setup_df)
                best_model = compare_models_cl()
                compare_df = pull_cl()
                st.info('This is the ML Model:')
                st.dataframe(compare_df)
                best_model
                save_model_cl(best_model, 'best_model')
            elif model == 'Regression':
                setup_rg(df, target=target, train_size=train_size/100, silent=True)
                setup_df_rg = pull_rg()
                st.info("This is the ML settings")
                st.dataframe(setup_df_rg)
                best_model = compare_models_rg()
                compare_df_rg = pull_rg()
                st.info('This is the ML Model:')
                st.dataframe(compare_df_rg)
                best_model
                save_model_rg(best_model, 'best_model')
            elif model == 'Clustering':
                setup_clust(df, silent=True)
                setup_df = pull_clust()
                st.info("This is the ML settings")
                st.dataframe(setup_df)
                kmeans = create_model('kmeans')

                compare_df = pull_clust()
                st.info('This is the Model - Kmeans:')
                st.dataframe(compare_df)

                result = assign_model(kmeans)
                st.info('Assigned clusters')
                st.dataframe(result)
                # best_model
                save_model_clust(kmeans, 'best_model_kmeans')

    if choice == "Download":
        st.title('Download the pkl file')
        with open('best_model.pkl', 'rb') as f:
            st.download_button('Download', f, "trained_model.pkl")

    if choice == "Prediction":
        st.title('Prediction for your values')

        if os.path.exists("best_model.pkl"):
            model = joblib.load('best_model.pkl')
            if st.checkbox('Enter values for one prediction'):

                target = st.selectbox('Select your target', df.columns)

                prediction_values = {}
                for namee in df.columns:
                    if namee != target:
                        prediction_values[namee] = st.text_input(namee)

                if st.button("Make prediction"):

                    # Use the pre-trained model to make a prediction
                    input_df = pd.DataFrame([list(prediction_values.values())], columns=list(prediction_values.keys()))
                    prediction = model.predict(input_df)

                    target_values = df[target].unique()
                    if all(str(ele).isdigit() for ele in target_values):
                        st.write('Prediction:', prediction[0])
                    else:
                        label_map = {code: name for code, name in enumerate(target_values)}
                        predicted_label = label_map[prediction[0]]
                        st.write('Prediction:', predicted_label)

            if st.checkbox('Upload a dataset for prediction'):
                file = st.file_uploader("Upload your dataset for prediction", type=['csv', 'txt', 'xls', 'xlsx'])
                if file:
                    input_df = pd.read_csv(file, index_col=None)
                    predictions = model.predict(input_df)
                    input_df['predictions'] = predictions

                    st.write('Predictions:')
                    st.write(input_df)

        else:
            st.write('Firstly, go to the ML page and train the models for classification or regression task')


if __name__:

    # -------- USER AUTHENTICATION --------


    # Show the login or sign up form based on the user's selection
    form_selection = st.sidebar.radio("Select an option", ["Login", "Sign Up"])
    if form_selection == "Login":
        users = db.fetch_all_users()

        usernames = [user['key'] for user in users]
        names = [user['name'] for user in users]
        hashed_passwords = [user['password'] for user in users]

        authentificator = stauth.Authenticate(names, usernames, hashed_passwords, 'autoMlapp', 'abcdef',
                                              cookie_expiry_days=30)

        name, authentication_status, username = authentificator.login("Login", "main")

        if authentication_status == False:
            print("Username/password is incorrect")
            st.error("Username/password is incorrect")

        if authentication_status == None:
            st.warning("Please enter your username and password")

        if authentication_status == True:
            print('User found! Successfully logged in')
            del form_selection
            main()

    elif form_selection == "Sign Up":
        db.delete_user('yanix')
        db.insert_user('yanix', 'Yana', 'qwerty')
        st.subheader("Create a new account")

        new_username = st.text_input("Username")
        new_name = st.text_input("Name")
        new_password = st.text_input("Password", type="password")

        if st.button("Submit"):
            db.insert_user(new_username, new_name, new_password)
            print("User created!")
            st.balloons()
            st.success("User created!")
