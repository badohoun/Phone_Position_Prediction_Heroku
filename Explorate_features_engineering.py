


import pandas as pd

import streamlit as st
import matplotlib.pyplot as plt
import shap
import numpy as np

import lightgbm as lgb

st.title('Phone Position Prediction app')


st.markdown("""
This app predicts the **Position Phone**


* **Python librairies:** pandas , streamlit , numpy , matplotlib , lightgbm.
* **Cloud : Azure Databricks : MLflow , Azure Machine Learning : AutoML
* **Data Source:** Azure Datalake  . Azure Storage Explorer
""")



st.sidebar.header('Specify Input Parameters')

uploaded_file = st.sidebar.file_uploader("Upload your input CSV file" , type = ["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        x_min = st.sidebar.slider('x_min', -1.8 , 0.8 , 0.16)
        x_max = st.sidebar.slider('x_max' , -0.07 , 2.3 , 0.5)
        x_std = st.sidebar.slider('x_std', 0.00049 , 0.6, 0.07)
        x_median = st.sidebar.slider('x_median', -0.9 , 0.9 , 0.3)
        y_min =  st.sidebar.slider('y_min', 4.29 , 0.1 , -0.18)
        y_max = st.sidebar.slider('y_max', -1.0 , 2.0 ,  0.24)
        y_std = st.sidebar.slider('y_std', 0.0004 , 2.0 ,  0.075)
        y_median = st.sidebar.slider('y_median', -1.09 , 1.01 ,  0.082)
        z_min = st.sidebar.slider('z_min', -2.05 , 1.0 ,  0.03)
        z_max = st.sidebar.slider('z_max', 0.53 ,3.0 ,  0.44)
        z_std = st.sidebar.slider('z_std', 0.0007 ,0.6 ,0.072)
        z_median = st.sidebar.slider('z_median', 0.6 ,1.0 ,0.2)
        norm_min = st.sidebar.slider('norm_min', 0.04 ,1.0 ,0.9)
        norm_max = st.sidebar.slider('norm_max', 0.9 ,4.4,1.0)
        norm_std = st.sidebar.slider('norm_std', 0.0007 ,0.9,0.07)
        norm_median = st.sidebar.slider('norm_median', 0.83 ,1.25,1.004)


        data = {'x_min':x_min,
                'x_max' : x_max,
                'x_std': x_std,
                'x_median':x_median,
                'y_min':y_min,
                'y_max':y_max,
                'y_std':y_std,
                'y_median':y_median,
                'z_min': z_min,
                'z_max': z_max,
                'z_std': z_std,
                'z_median': z_median,
                'norm_min': norm_min,
                'norm_max': norm_max,
                'norm_std': norm_std,
                'norm_median':norm_median}
        features = pd.DataFrame(data , index = [0])
        return features
df1 = user_input_features()

def load_data1():
   accelerometer_data_path = 'matrix_trajet_resample_5s.csv'
   data = pd.read_csv(accelerometer_data_path)
   return data

df2 =  load_data1()
st.subheader('User Input Parameters')

if uploaded_file is not None :
    st.write(df1)
else:
    st.write('Awaiting CSV file to be uploaded . Currently using example input parameters ')
    st.write(df1)



st.subheader('Training')

st.write(df2.head(5))




X = df2.drop(labels = ['activity' , 'datetime'], axis =1)
Y = df2.activity

model = lgb.LGBMClassifier(n_estimators= 6000, random_state=np.random.seed(seed=123) , n_jobs=10, num_leaves=100  , objective = 'multiclass' , boosting_type = 'dart')

model.fit(X, Y)


explainer = shap.TreeExplainer(model)
shap_values =explainer.shap_values(X)

st.header('Feature Importance')
plt.title('Feature importance based on SHAP values (Bar) ')

shap.summary_plot(shap_values , X , plot_type="bar")

st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot(bbox_inches = 'tight')


# Apply a model to make prediction

prediction = model.predict(df1)





st.header('Prediction of Activity : fixe , siege , poche intérieure')

st.write(prediction)
