import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from util import *
import warnings
warnings.filterwarnings("ignore")

#get dataset
df = pd.read_csv("heart_disease_data.csv")

y = df['target']
x = df.drop(['target'],axis=1)
import statsmodels.api as sm

reg_log = sm.Logit(y,x)
res = reg_log.fit()


np.set_printoptions(formatter={'float':lambda x: "{0:0.2f}".format(x)})


cm = confusion_matrix(x,y,res) 
print('Metrics and accuracy: ')
print(cm)

#single prediction
ip = (62,0,0,140,268,0,0,160,0,3.6,0,2,2)

ip_data = np.asarray(ip)

#getting probability
print('Chances of heart disease: ',round(res.predict(ip_data)[0]*100,2))