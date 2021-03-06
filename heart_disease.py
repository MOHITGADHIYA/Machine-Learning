**IMPORTING LIBRARIES**

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

"""**DATA COLLECTION**"""

df=pd.read_csv('/content/drive/My Drive/Dataset/heart.csv')

df.head(3)

"""**DATA ANALYSIS AND PREPROCESSING**"""

def load_from_csv(file_name):
  with open(file_name, 'r') as file:
    reader = file.read()
    reader.split(" ")
    data = []
    for row in reader.split("\n"):
      tmp = []
      for values in row.split(","):
        tmp.append(values)
      data.append(tmp)
  return data 
 


rows=load_from_csv('/content/drive/My Drive/Dataset/heart.csv')

col=""
col_1=[]
col_2=""
n=0
for i in df.columns:
  col=i
  col_1.append(col)
#print(col)
#print(col_1) 
column=col_2.join(col_1).split("\t")
print(column)

new_df=pd.DataFrame(columns=column)

new_df

data_1=[]
data_2=""
for row in range(1,304):
  data_1.append(rows[row])
  for i in data_1:
    data_3=data_2.join(i).split("\t")#print(data4)
  a_series = pd.Series(data_3, index = new_df.columns)
  new_df = new_df.append(a_series, ignore_index=True)

new_df

new_df.info()

for col in new_df.columns:
  new_df[col]=pd.to_numeric(new_df[col])
new_df.info()

new_df.isnull().sum()

"""**DATA VISUALIZATION**"""

plt.figure(figsize=(10,10))
sns.heatmap(new_df.corr(),vmax=1,vmin=-1,cmap="Blues",annot=True)

x=new_df.drop(["target","bps","fasting_blood_suger","resting electrocardiographic"],axis=True)

y=new_df.target

"""**SPLITIND DATA FOR TRAINING AND TESTING**"""

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

from sklearn.preprocessing import StandardScaler
SS=StandardScaler()
x_train=SS.fit_transform(x_train)
x_test=SS.fit_transform(x_test)

"""**MODELING**

**RANDOM FORESR CLASSIFIER**
"""

from sklearn.ensemble import RandomForestClassifier
rfs=RandomForestClassifier()
rfs.fit(x_train,y_train)
rfs.predict(x_test)
rfs.score(x_test,y_test)

"""**DICISION TREE CLASSIFIER**"""

from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()
dtc.fit(x_train,y_train)
dtc.predict(x_test)
dtc.score(x_test,y_test)

"""**LOGISTIC REGRESSION**"""

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)
lr.predict(x_test)
lr.score(x_test,y_test)

"""**KNN**"""

from sklearn.neighbors import KNeighborsClassifier
kn=KNeighborsClassifier(n_neighbors=3)
kn.fit(x_train,y_train)
kn.predict(x_test)
kn.score(x_test,y_test)
