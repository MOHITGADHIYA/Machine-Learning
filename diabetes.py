"""**IMPORTING LIBRARIES**"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""**DATA COLLECTION**"""

dataset=pd.read_csv('/content/drive/My Drive/Dataset/diabetes.csv')

dataset.head()

dataset.shape

"""**DATA ANALYSIS**"""

dataset.describe

dataset.info()

dataset.isnull().sum()

dataset["class"].value_counts()

"""**PREPROCESSING**"""

from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
dataset["class"]=LE.fit_transform(dataset["class"])

"""**VISUALIZATION**"""

sns.heatmap(dataset.corr(),vmax=1,vmin=-1,annot=True,cmap="Blues")

x=dataset.drop(["class"],axis=True)
y=dataset["class"]

"""**SPLITING THE DATA FOR TRAINING AND TESTING**


"""

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

from sklearn.preprocessing import StandardScaler
SS=StandardScaler()
x_train = SS.fit_transform(x_train)
x_test = SS.fit_transform(x_test)

"""**MODELING**

**SUPPORT VECTOR MACHINE**
"""

from sklearn.svm import SVC
model=SVC()
model.fit(x_train,y_train)
model.predict(x_test)
model.score(x_test,y_test)
