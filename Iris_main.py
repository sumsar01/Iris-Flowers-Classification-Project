import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
import sqlite3
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from numpy.random import seed
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from sklearn.preprocessing import LabelEncoder

rcParams['figure.figsize'] = 10,8
sns.set(style='whitegrid', palette='bright',
        rc={'figure.figsize': (15,10)})

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


#creating model
def create_model(lyrs=[8], act='linear', opt='Adam', dr=0.0):
    
    #set random seed
    seed(1)
    tf.random.set_seed(1)
    
    model = Sequential()
    
    #create first hidden layer
    model.add(Dense(lyrs[0], input_dim=X_train.shape[1], activation=act))
    
    #create additional hidden layers
    for i in range(1, len(lyrs)):
        model.add(Dense(lyrs[i], activation=actlen))
        
    #add dropout, default is none
    model.add(Dropout(dr))
    
    #create output layer
    model.add(Dense(1, activation='sigmoid'))
    
    
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    return model

#creating first model
model = create_model()
print(model.summary())

training = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)
val_acc = np.mean(training.history['val_accuracy'])
print("\n%s: %.2f%%" % ('val_accuracy', val_acc*100))























