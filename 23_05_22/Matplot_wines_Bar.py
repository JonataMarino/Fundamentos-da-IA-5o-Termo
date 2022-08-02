import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#plt.rcParams["figure.figsize"] = [7.00, 3.50]
#plt.rcParams["figure.autolayout"] = True
columns = ['alcohol', 'quality']
df = pd.read_csv("E:/logatti/Fundamentos da inteligencia artificial/23_05_22/csv/wine_dataset.csv", usecols=columns)
#print ('\n', df)
print(df.iloc[:,[1]])
#print(df.groupby('quality')['alcohol'].count())
#print(df.groupby('quality')['alcohol'].mean())
matriz = df.groupby('quality',as_index=False)['alcohol'].mean()
print(matriz.iloc[:,[0]])
eixo_x = str(matriz.iloc[:,[0]])
eixo_y = str(matriz.iloc[:,[1]])
#the_arr = np.array(matriz)
#print(matriz[:, 1])
plt.bar(eixo_y, eixo_x , label = 'quality/alcohol')
#plt.xlabel('Quality')
#plt.ylabel('Alcohol')
#plt.bar(x_axis, y_axis)

#print(the_arr[:,[0]])
plt.show()