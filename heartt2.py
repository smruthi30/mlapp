# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 18:03:56 2023

@author: DELL
"""
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from streamlit_option_menu import option_menu

data=pd.read_csv('C:/Users/DELL/Desktop/st1/heart.csv')

#heart_disease_model = pickle.load(open('C:/Users/DELL/Desktop/st1/heart_disease_model.sav','rb'))
rfbm_model = pickle.load(open('C:/Users/DELL/Desktop/st1/heart_disease_model_rfbm.sav','rb'))
dtbm_model = pickle.load(open('C:/Users/DELL/Desktop/st1/heart_disease_model_dtbm.sav','rb'))
adaboost_model = pickle.load(open('C:/Users/DELL/Desktop/st1/heart_disease_model_adaboost.sav','rb'))
gb_model = pickle.load(open('C:/Users/DELL/Desktop/st1/heart_disease_model_gradientboost.sav','rb'))
knn_model = pickle.load(open('C:/Users/DELL/Desktop/st1/heart_disease_model_knnbm.sav','rb'))
rfbm_r = pickle.load(open('C:/Users/DELL/Desktop/st1/relief_rfbm.sav','rb'))
dtbm_r = pickle.load(open('C:/Users/DELL/Desktop/st1/relief_dtbm.sav','rb'))
adaboost_r = pickle.load(open('C:/Users/DELL/Desktop/st1/relief_adaboost.sav','rb'))
gb_r = pickle.load(open('C:/Users/DELL/Desktop/st1/relief_gradientboost.sav','rb'))
knn_r = pickle.load(open('C:/Users/DELL/Desktop/st1/relief_knnbm.sav','rb'))

    
# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('Efficient prediction of cardiovascular disease using ML algorithms with relief and lasso feature selection ',
                          
                          ['Home','Data visualisation',
                           'Heart Disease Prediction'],
                          icons=['house','bar-chart-line','activity',],
                          default_index=0)
if (selected == 'Home'):
    st.title("Use of Relief and LASSO Feature Selection Approaches in Machine Learning Algorithms for Effective Cardiovascular")
    image=Image.open("C:/Users/DELL/Desktop/st1/dataset-cover.jpg")
    st.image(image, caption='',output_format="auto")
    '''One of the most prevalent and significant diseases affecting peoples health is
  cardiovascular disease (CVD). Early diagnosis may allow for CVD mitigation or prevention,
which could lower mortality rates. A viable strategy is to find risk factors using machine
learning algorithms. We would like to suggest a model that combines various approaches
to obtain accurate cardiac disease prediction. We have successfully created accurate data
for the training model using effective approaches for Data Collecting, Data
Pre-processing, and Data Transformation.
A merged dataset was deployed (Cleveland, Long Beach VA, Switzerland, Hungarian and
Stat log). The Relief and Least Absolute Shrinkage and Selection Operator (LASSO)
approaches are used to choose suitable features. By combining the traditional classifiers
with bagging and boosting methods (which are used in the training process) and new
hybrid classifiers like Decision Tree Bagging Method (DTBM), Random Forest Bagging
Method (RFBM), K-Nearest Neighbors Bagging Method (KNNBM), AdaBoost Boosting
Method (ABBM), and Gradient Boosting Boosting Method (GBBM) are created.'''
# Heart Disease Prediction Page
if (selected == 'Heart Disease Prediction'):
    
    # page title
    st.title('Heart Disease Prediction using ML')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age')
    with col2:
        sex = st.text_input('Sex')
        
    with col3:
        cp = st.text_input('Chest Pain types')
        
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
        
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')
        
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
        
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')
        
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
        
    with col3:
        exang = st.text_input('Exercise Induced Angina')
        
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')
        
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')
        
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')
        
    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
# code for Prediction
    #heart_diagnosis = ''
    
    # creating a button for Prediction
    fs = st.radio(
    "What\'s type of feature selection you want",
    ('Lasso', 'Relief',))
    st.write('You selected:', fs)


    option=st.selectbox(
    'choose the model u want to use for testing',
    ( 'Decision tree bagging method(DTBM)','Random Forest bagging method(RFBM)','K-Nearest Neighbors Bagging Method (KNNBM),','AdaBoost Boosting Method (ABBM)', 
     'GradientBoosting Boosting Method (GBBM)'))
    st.write('You selected:', option)
    
    if st.button('Heart Disease Test Result'):
        #heartd_prediction = heartd_prediction.predict([[age, sex, cp, trestbps, chol, fbs, restecg,thalach,exang,oldpeak,slope,ca,thal]])                          
        input_data_as_numpy_array=np.array([age, sex,trestbps, chol, fbs, restecg,thalach,oldpeak,slope,ca,thal],dtype=int)
        
         # reshape the numpy array as we are predicting for only on instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
        if option=='Decision tree bagging method(DTBM)':
            prediction = dtbm_model.predict(input_data_reshaped)
            if (prediction== 0):
               st.success('The Person does not have a Heart Disease')
          #st.write('The Person does not have a Heart Disease')
            else:
              st.success('The Person has Heart Disease')
          #st.write('The Person has Heart Disease')
        elif option=='Random Forest bagging method(RFBM)':
                prediction = rfbm_model.predict(input_data_reshaped)
                if (prediction== 0):
                   st.success('The Person does not have a Heart Disease')
                else:
                   st.success('The Person has Heart Disease')
        elif option=='K-Nearest Neighbors Bagging Method (KNNBM)':
                prediction = knn_model.predict(input_data_reshaped)
                if (prediction== 0):
                   st.success('The Person does not have a Heart Disease')
                else:
                   st.success('The Person has Heart Disease')
        elif option=='AdaBoost Boosting Method (ABBM)':
           prediction = adaboost_model.predict(input_data_reshaped)
           if (prediction== 0):
              st.success('The Person does not have a Heart Disease')
           else:
              st.success('The Person has Heart Disease')
        elif option=='GradientBoosting Boosting Method (GBBM)':
           prediction = gb_model.predict(input_data_reshaped)
           if (prediction== 0):
              st.success('The Person does not have a Heart Disease')
           else:
               st.success('The Person has Heart Disease')  
        
    
if (selected == 'Data visualisation'):
    st.title('Data visualisation')

    
    if st.checkbox("display dataset"):
        st.dataframe(data.head(10))
    if st.checkbox("display correlation"):  
        cor=data.corr()
        top_corr_feature=cor.index
        fig = plt.figure(figsize=(10,10))
        sns.heatmap(data[top_corr_feature].corr(),annot=True,cmap="RdYlGn")
        st.title("HEATMAP")
        st.pyplot(fig)
    if st.checkbox("bargraph visualisation of data"):
        fig1 = plt.figure(figsize=(3, 3))
        plt.title("count of male and female")
        
        sns.countplot(x="sex", data=data)
        fig2 = plt.figure(figsize=(3, 3))
        plt.title("count of people with disease in various stages")
        sns.countplot(x="target", data=data)
        
        # 1 is male
        # 0 is female
        fig3 = plt.figure(figsize=(3, 3))
        sns.countplot(x="target", data=data,hue = 'sex')
        plt.title("GENDER - Heart Diseases")
        col1, col2 ,col3 = st.columns(3)
        
        with col1:
          st.pyplot(fig1)
        with col2:
            st.pyplot(fig2)
        with col3:
            st.pyplot(fig3)
    if st.checkbox("histogram visualisation of data"):
        fig4=plt.figure(figsize=(18,18))
        ax=fig4.gca()
        data.hist(ax=ax,bins=30)
        st.pyplot(fig4)
    if st.checkbox("Analysing each features"):
        fig1 = plt.figure(figsize=(3, 3))
        temp = (data.groupby(['target']))['cp'].value_counts(normalize=True)\
        .mul(100).reset_index(name = "percentage")
        sns.barplot(x = "target", y = "percentage", hue = "cp", data = temp)
        plt.title("chestpain vs heart disease")
        
        fig2 = plt.figure(figsize=(3, 3))
        temp = (data.groupby(['target']))['fbs'].value_counts(normalize=True)\
        .mul(100).reset_index(name = "percentage")
        sns.barplot(x = "target", y = "percentage", hue = "fbs", data = temp)
        plt.title("fbs vs target")
        
        fig3 = plt.figure(figsize=(3, 3))
        temp = (data.groupby(['target']))['restecg'].value_counts(normalize=True)\
        .mul(100).reset_index(name = "percentage")
        sns.barplot(x = "target", y = "percentage", hue = "restecg", data = temp)
        plt.title("resting electrocardiographic results vs Heart Disease")
        
  
        
        fig = plt.figure(figsize=(3, 3))
        temp = (data.groupby(['target']))['ca'].value_counts(normalize=True)\
        .mul(100).reset_index(name = "percentage")
        sns.barplot(x = "target", y = "percentage", hue = "ca", data = temp)
            
        plt.title("ca vs Heart Disease")
       
        
        fig5=plt.figure(figsize=(3, 3))
        temp = (data.groupby(['target']))['thal'].value_counts(normalize=True)\
        .mul(100).reset_index(name = "percentage")
        sns.barplot(x = "target", y = "percentage", hue = "thal", data = temp)
        plt.title("thal vs Heart Disease")
       
        col1, col2,col3  = st.columns(3)
        with col1:
            st.pyplot(fig1)
        with col2:
            st.pyplot(fig2)
        with col3:
            st.pyplot(fig3)
        with col1:
            st.pyplot(fig)
        with col2:
            st.pyplot(fig5)
              
            
        
    if st.checkbox("pairplot"):
        fig=sns.pairplot(data)
        fig.figsize=(20, 20)
        st.pyplot(fig)
        