import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
import sqlite3
from sklearn.preprocessing import StandardScaler

rcParams['figure.figsize'] = 10,8
sns.set(style='whitegrid', palette='bright',
        rc={'figure.figsize': (15,10)})

#Getting data from SQLite database
db_file = "./Data/database.sqlite"
con = sqlite3.connect(db_file)
df = pd.read_sql_query("SELECT * FROM Iris", con, index_col="Id")
con.close()



print(df.head())
"""
#Species ['Iris-setosa' 'Iris-versicolor' 'Iris-virginica']
#Plotting data
ax = sns.jointplot(data=df, x="SepalLengthCm", y="SepalWidthCm", hue="Species", kind="kde")
ax.set_axis_labels(xlabel='Sepal Length (Cm)', ylabel='Sepal Width (Cm)')

ax = sns.jointplot(data=df, x="PetalLengthCm", y="PetalWidthCm", hue="Species", kind="kde")
ax.set_axis_labels(xlabel='Petal Length (Cm)', ylabel='Petal Width (Cm)')
"""

#Re-scaling continuous parameters
continuous = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

scaler = StandardScaler()

for var in continuous:
    df[var] = df[var].astype('float64')
    df[var] = scaler.fit_transform(df[var].values.reshape(-1,1))
    
#test-train split