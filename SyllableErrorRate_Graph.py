import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

x_values = [30, 45, 75, 180, 600]
y_values = {
    30: [5.36, 7.24, 7.06, 5.64, 4.78],
    45: [3.63, 5.33, 4.28, 4.99, 4.96],
    75: [4.02, 3.59, 2.61, 3.59, 3.69],
    180: [3.52, 2.59, 2.5, 2.85, 3.04],
    600: [2.43, 2.29, 1.91, 2.66, 3.26]
}

means = np.array([np.mean(y_values[x]) for x in x_values])
std_devs = np.array([np.std(y_values[x]) for x in x_values])

x_all = np.array([x for x in x_values for _ in y_values[x]])
y_all = np.array([y for sublist in y_values.values() for y in sublist])

interp_linear = interp1d(x_values, means, kind='linear')

x_new = np.linspace(min(x_values), max(x_values), 300)
y_new = interp_linear(x_new)

interp_std_dev = interp1d(x_values, std_devs, kind='linear')
std_dev_new = interp_std_dev(x_new)

plt.figure(figsize=(10, 6))

plt.scatter(x_all, y_all, color='gray', alpha=0.6)

plt.errorbar(x_values, means, yerr=std_devs, fmt='o', color='red', ecolor='lightgray', elinewidth=3, capsize=5, label='Moyenne')

plt.plot(x_new, y_new, '-', color='blue')

plt.fill_between(x_new, y_new - std_dev_new, y_new + std_dev_new, color='lightblue', alpha=0.4, label='Écart-type')

plt.xticks(x_values)

plt.title('Tweetynet syllable error rate')
plt.xlabel('Training set duration (s)')
plt.ylabel('Syllable error rate (%)')
plt.legend()
plt.grid(True)
plt.show()
