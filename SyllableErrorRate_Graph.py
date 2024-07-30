import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

x = np.array([30, 75, 180, 600])
y = np.array([6.016, 3.805, 2.9, 2.21])

# Interpolation linéaire
interp_linear = interp1d(x, y, kind='linear')

x_new = np.linspace(x.min(), x.max(), 300)
y_new = interp_linear(x_new)

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'o')
plt.plot(x_new, y_new, '-')  # Courbe

plt.xticks(x)  # Assure que les graduations correspondent exactement aux valeurs de x

plt.title('Tweetynet syllable error rate')
plt.xlabel('Training set duration (s)')
plt.ylabel('Syllable error rate (%)')
plt.legend()
plt.grid(True)
plt.show()

