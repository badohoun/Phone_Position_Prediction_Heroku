


import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import pickle5 as pickle
import time
import os

os.chdir("c:\\Users\\inno-demo\\Downloads\\ML_Model_for_Predicting_Phone_Position_in_car\\Algo_Filtrage_Position_Telephone_en_Voiture\\Streamlit\\Accelerometer\\Model_Building")
import pandas as pd
matrix_trajet_resample_5 = pd.read_csv('matrix_trajet_resample_5s.csv')






#from sklearn.preprocessing import StandardScaler
#scaler1 = StandardScaler()
#scaler1.fit(matrix_trajet_resample_5.drop(labels = ['activity'] , axis = 1))
#features_res_5 = scaler1.transform(matrix_trajet_resample_5.drop(labels = ['activity'] , axis = 1))




# 2. Set `python` built-in pseudo-random generator at a fixed value










features_res_5= matrix_trajet_resample_5.drop(labels = ['datetime', 'activity'] , axis = 1)

target_res_5  = matrix_trajet_resample_5.activity



import numpy as np

np.random.seed(42)
learning_rate = 0.46
is_unbalance=True
n_estimators=6000
n_jobs=1
num_class=3
num_leaves=90
objective = 'multiclass'
boosting_type = 'dart'

ts = time.time()
#!pip install lightgbm
import lightgbm as lgb
model = lgb.LGBMClassifier(learning_rate = learning_rate , is_unbalance=is_unbalance, n_estimators=n_estimators, n_jobs=n_jobs, num_class=num_class,num_leaves=num_leaves, objective = objective, boosting_type = boosting_type)

model.fit(features_res_5, target_res_5)



# saving the model
!pip install pickle5
import pickle5 as pickle
pickle.dump(model,open('bestlightgbm2_model.pkl' , 'wb'))
