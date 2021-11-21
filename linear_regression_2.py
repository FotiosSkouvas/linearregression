#import essential libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#Statement of our multi-element containers
header = st.container()
dataset = st.container()
data_overview = st.container()
data_visualization = st.container()
data_quering = st.container()
data_modeling = st.container()

#Introduction
with header:
    st.title('Predictive Maintenance')
    st.markdown('**Sooner or Later, all machines run to a failure!**')
    st.markdown('Predictive Maintenance is a type of condition based maintenance where maintenance is only scheduled when specific conditions are met and before the equipment breaks down.')
#Insert of dataset
with dataset:
    st.header('Dataset')
    st.markdown('Using this application, you will have the opportunity to analyze your dataset and find estimate when the maintence should be performed to repair the damaged parts')
    uploaded_file = st.file_uploader("First of all, choose a CSV file:")
    if uploaded_file is not None:
        st.write('To ensure that you have provided the right dataset, below you will find the first five rows of your dataset: ')
        pred_data = pd.read_csv(uploaded_file)
        st.write(pred_data.head(5))

        #Dataset Overview
        with data_overview:
            st.header('Dataset Overview')
            st.markdown('Now you will have the oportunity to see the structure of your dataset:')
            st.markdown('*Your dataset contains the following columns:*')
            st.write(pred_data.columns)
            st.markdown('You will now have the opportunity to check if your dataset contains outliers.')
            st.markdown('*The table below provides a description of your dataset:*')
            st.write(pred_data.describe())
            st.markdown('The non NULL analysis provided below, will help you check if there are missing values in your dataset.')
            st.markdown('*Non NULL values analysis:*')
            st.write(pred_data.count())
            st.markdown('Here, we should provide a description of the values of the dataset, in order to check the dataset for format issues, but there is a problem with dtypes function at streamlit')
            st.header('Data Cleaning')
            clean = st.radio(
            'Is your data clean?',
            ('Yes', 'No'))
            if clean == 'Yes':
                st.success('You can proceed with the analysis')

                #Data visualization options
                with data_visualization:
                    st.header('Data Visualization')
                    st.markdown('In this section you will have the opportunity to visualize your dataset')
                    opt_vis = st.radio(
                    'Please select your preferred plotting option: ',
                    ('Scatter Plot', 'Pie Chart'))
                    if opt_vis == 'Scatter Plot':
                        x_axis = st.selectbox('Please select x axis:',
                        pred_data.columns)
                        y_axis = st.selectbox('Please select y axis:',
                        pred_data.columns)
                        st.markdown('*Scatter Plot*')
                        fig_1 = px.scatter(pred_data, x = pred_data[x_axis], y = pred_data[y_axis])
                        st.plotly_chart(fig_1)
                    else:
                        st.markdown('*Pie chart*')
                        val_1 = st.selectbox('Please select the pie chart value:',
                        pred_data.columns)
                        fig_2, ax = plt.subplots()
                        ax.pie(pred_data[val_1].value_counts(), labels = pred_data[val_1].unique(), autopct='%1.1f%%')
                        ax.set_title(val_1)
                        st.pyplot(fig_2)

                #Data quering
                with data_quering:
                    st.header('Data Quering')
                    st.markdown('You will now have the opportunity to identify the contribution of a selected failure and a possible selected cause')
                    failure = st.selectbox('Please select a possible failure of your system:',
                    pred_data.columns)
                    pred_data_root = pred_data[pred_data[failure]==1]
                    st.write('Total number of this category failures in your database: ', len(pred_data_root))
                    percentage = len(pred_data_root)/len(pred_data)*100
                    st.write(percentage, '% of observations failed due to this failure category')
                    st.markdown('The contribution of which feature would you like to observe for the selected root cause?')
                    feat = st.selectbox('Please select a possible cause of your system:',
                    pred_data.columns)
                    pred_data_feat = pred_data_root[feat].value_counts()
                    st.write(pred_data_feat)

                #Data modeling
                with data_modeling:
                    st.header('Model Training')
                    st.markdown('*Our application uses linear regression method between the selecet failure and the selected cause. In order to reduce to error, the application uses the dataset that the selected failure = 1*')
                    x_train, x_test, y_train, y_test = train_test_split(pred_data_root[failure], pred_data_root[feat], test_size = 0.2)
                    regr = LinearRegression()
                    regr.fit(np.array(x_train).reshape(-1,1), y_train) #Reshape transverses it from a single dimension matrix to a vertical shape
                    preds = regr.predict(np.array(x_test).reshape(-1,1))
                    st.write('The expected mean value of feature is ', regr.intercept_)
                    y_pred = regr.predict(np.array(x_test).reshape(-1,1))

                    #Evaluation of model
                    st.markdown('Evaluation of the model')
                    st.write('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
                    st.write('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
                    st.write('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
            else:
                st.error('Please clean your data and upload the updated CSV file to continue')
    else:
        st.error('Please upload a CSV file to continue')
