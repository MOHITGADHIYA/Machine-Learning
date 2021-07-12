# -*- coding: utf-8 -*-
"""KNN_algo.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1TN35R9uMMRsAtE4dOSzlSg3E1gdB3kWN
"""

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd

df=pd.read_csv("/content/drive/My Drive/Dataset/iris_csv.csv")
df

from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
df["class"]=LE.fit_transform(df["class"])

x=df.drop(["class"],axis=True)
y=df["class"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()

x_train=scaler.fit_transform(x_train)
x_test=scaler.fit_transform(x_test)

from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier()
model.fit(x_train,y_train)
model.predict(x_test)
model.score(x_test,y_test)

model.score(x_test,y_test)

# from sklearn.metrics import accuracy_score

# print(accuracy_score(prdiction,y_test))

# from  sklearn.cluster import KMeans

# km=KMeans(n_clusters=3)
# km

# y_predected=km.fit_predict(df[['petallength','petalwidth']])

# y_predected

# df['cluster']=y_predected
# df

# df1=df[df.cluster==0]
# df2=df[df.cluster==1]
# df3=df[df.cluster==2]

# plt.scatter(df1[['petallength']],df1[['petalwidth']],color="green")
# plt.scatter(df2[['petallength']],df2[['petalwidth']],color="red")
# plt.scatter(df3[['petallength']],df3[['petalwidth']],color="yellow")

# plt.xlabel('petallength')
# plt.ylabel('petalwidth')
# plt.legend()

# sse=[]
# for k in range(1,10):
#   km=KMeans(n_clusters=k)
#   km.fit(df[['petallength']],df[['petalwidth']])
#   sse.append(km.inertia_)

# plt.xlabel('K')
# plt.ylabel('Sum of squared error')
# plt.plot(range(1,10),sse)

