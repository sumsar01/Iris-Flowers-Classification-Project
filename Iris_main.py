import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
import sqlite3

rcParams['figure.figsize'] = 10,8
sns.set(style='whitegrid', palette='bright',
        rc={'figure.figsize': (15,10)})

#Getting data from SQLite database
db_file = "./Data/database.sqlite"
con = sqlite3.connect(db_file)
df = pd.read_sql_query("SELECT * FROM Iris", con)
con.close()



print(df.head())