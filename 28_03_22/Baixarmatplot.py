import matplotlib.pyplot as plt

import numpy as np

#prepare the data

x = np.linspace(0, 10, 100, 200)

#plot the data

plt.plot(x, x, label = 'linear')

#add a legend

plt.legend()


#show the plot

plt.show()
