import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
import sqlite3
from numpy.random import seed
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn import svm

rcParams['figure.figsize'] = 10,8
sns.set(style='whitegrid', palette='bright',
        rc={'figure.figsize': (15,10)})

#If using neural network model set 1
with_NN = 1
seed(1)

#Getting data from SQLite database
db_file = "./Data/database.sqlite"
con = sqlite3.connect(db_file)
df = pd.read_sql_query("SELECT * FROM Iris", con, index_col="Id")
con.close()


 

"""
#Species ['Iris-setosa' 'Iris-versicolor' 'Iris-virginica']
#Plotting data
ax = sns.jointplot(data=df, x="SepalLengthCm", y="SepalWidthCm", hue="Species", kind="kde")
ax.set_axis_labels(xlabel='Sepal Length (Cm)', ylabel='Sepal Width (Cm)')

ax = sns.jointplot(data=df, x="PetalLengthCm", y="PetalWidthCm", hue="Species", kind="kde")
ax.set_axis_labels(xlabel='Petal Length (Cm)', ylabel='Petal Width (Cm)')
"""

#Re-scaling continuous parameters
if with_NN == 1:
    continuous = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    scaler = StandardScaler()
    for var in continuous:
        df[var] = df[var].astype('float64')
        df[var] = scaler.fit_transform(df[var].values.reshape(-1,1))



#Label encoding target 'Iris-setosa'=0 'Iris-versicolor'=1 'Iris-virginica'=2
label_encoder = LabelEncoder()
df['Species'] = label_encoder.fit_transform(df['Species'])

    
print(df.head())    

#test-train split
X_train, X_test, y_train, y_test = train_test_split(df.drop(['Species'], axis=1), df['Species'], 
                                                    test_size=0.10, random_state=1)


model = svm.SVC()
model.fit(X_train, y_train)


cross_val = cross_val_score(model, X_train, y_train, cv=5)
print( "%f%% is the result for the first model\n" % (cross_val.mean()))

#We now tune parameters
kernel = ['linear', 'poly', 'rbf', 'sigmoid']
param_grid = dict(kernel=kernel)

#We now want to use grid search
grid = GridSearchCV(estimator=model,
                    param_grid=param_grid,
                    cv=5,
                    verbose=2,
                    n_jobs=-1)

grid_result = grid.fit(X_train, y_train)

print("\nThe best result was with")
print(grid_result.best_params_)
print("and had a precision of %f%%, this is a improvement of %f%%\n" 
      %(grid_result.best_score_, grid_result.best_score_-cross_val.mean()))

first_opt = grid_result.best_score_





