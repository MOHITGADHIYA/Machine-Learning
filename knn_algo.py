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
