import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
data={
    'name':['john','anna','peter','linda','james','np.nan'],
    'age':[26,22,np.nan,32,45,36],
    'salary':[50000,54000,58000,np.nan,62000,60000],
    'department':['hr','it','finance','it','finance','hr']
    }
df=pd.DataFrame(data)
print("original dataset:\n", df)
df['age'].fillna(df['age'].mean(), inplace=True)
df['salary'].fillna(df['salary'].median(), inplace=True)
df['name'].fillna("deepu", inplace=True)
print("\n After handling missing values:\n",df)
label_encoder = LabelEncoder()
df['dept_label'] = label_encoder.fit_transform(df['department'])
print("\n after label encoding:\n", df)
df=pd.get_dummies(df,columns=['department'])
print("\n after one-hot encoding:\n", df)
scaler = MinMaxScaler()
df[['age', 'salary']] = scaler.fit_transform(df[['age', 'salary']])
std_scaler=StandardScaler()
df[['age','salary']]= std_scaler.fit_transform(df[['age','salary']])
print("\n after feature scaling:\n",df)
x=df.drop(['name'],axis=1)
y=df['name']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
print("\n training features:\n",x_train)
print("\n testing features:\n",x_test)
print("\n training features:\n",y_train)
print("\n testing features:\n",y_test)
                                        
