import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""**DATA COLLECTION**"""

dataset=pd.read_csv('D:/Dataset/winequality.csv')

dataset

"""**DATA ANALYSIS AND PREPROCESSING**"""

em_list=[]
em_str=""
for i in dataset.columns:
  em_list.append(i)

em_str=em_str.join(em_list).replace('"',"")

column=em_str.split(";")
column

new_dataset=pd.DataFrame(columns=column)
new_dataset

def load_from_csv(file_name):
  with open(file_name, 'r') as file:
    reader = file.read()
    reader.split(" ")
    data = []
    for row in reader.split("\n"):
      tmp = []
      for values in row:
        tmp.append(values)
      data.append(tmp)
  return data 
 


rows=load_from_csv('/content/drive/My Drive/Dataset/winequality.csv')

data_1=[]
em_str1=""
for row in range(1,1600):
  data_1.append(rows[row])
  for i in data_1:
    row_=em_str1.join(i).split(";")
    a_series = pd.Series(row_, index = new_dataset.columns)
  new_dataset = new_dataset.append(a_series, ignore_index=True)

new_dataset.head()

new_dataset.info()

for i in new_dataset.columns:
  new_dataset[i]=pd.to_numeric(new_dataset[i])

new_dataset.describe()

new_dataset.isnull().sum()

"""**DATA VISUALIZATION**"""

plt.figure(figsize=(10,10))
sns.heatmap(new_dataset.corr(),annot=True,vmin=-1,vmax=1,cmap="Blues")

x=new_dataset.drop(["quality","fixed acidity","residual sugar","free sulfur dioxide","total sulfur dioxide","pH"],axis=True)
y=new_dataset["quality"]

"""**SPLITING THE DATA FOR TRAINING AND TESTING**"""

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

"""**BUILDING A MODEL** \
**RANDOM FOREST CLASSIFIER**
"""

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=100)

model.fit(x_train,y_train)

model.predict(x_test)

print("Training accuracy :", model.score(x_train, y_train))
print("Testing accuracy :", model.score(x_test, y_test))
