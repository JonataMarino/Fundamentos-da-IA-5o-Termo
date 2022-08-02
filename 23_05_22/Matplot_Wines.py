import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
columns = ['alcohol', 'quality']
df = pd.read_csv("E:/logatti/Fundamentos da inteligencia artificial/23_05_22/csv/wine_dataset.csv", usecols=columns)
print ('\n', df)

#print(df.groupby('quality')['alcohol'].count())
print(df.groupby('quality')['alcohol'].mean())


plt.plot(df.groupby('quality')['alcohol'].mean())
plt.show()