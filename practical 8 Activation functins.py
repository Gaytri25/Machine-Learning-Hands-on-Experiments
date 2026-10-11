
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.DataFrame({'Input': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]})
x = data['Input']

# Calculate A Functions
data['Sigmoid'] = 1 / (1 + np.exp(-x))
data['Tanh'] = np.tanh(x)
data['ReLU'] = np.maximum(0, x)
data['Leaky ReLU'] = np.where(x > 0, x, 0.01 * x)

print(data)


plt.plot(x, data['Sigmoid'], label='Sigmoid')
plt.plot(x, data['Tanh'], label='Tanh')
plt.plot(x, data['ReLU'], label='ReLU')
plt.plot(x, data['Leaky ReLU'], label='Leaky ReLU')

plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Activation Functions')
plt.legend()
plt.grid()
plt.show()



#  Plot Activation Functions  on dataset iries 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Step 2: Select one real feature
x = df['sepal length (cm)']

# Step 3: Calculate activation functions
sigmoid = 1 / (1 + np.exp(-x))
tanh = np.tanh(x)
relu = np.maximum(0, x)
leaky_relu = np.where(x > 0, x, 0.01 * x)

# Step 4: Display the dataset
print("Iris Dataset:")
print(df.head())

# Step 5: Plot A Functions
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.plot(x, sigmoid)
plt.title("Sigmoid Function")
plt.xlabel("Sepal Length")
plt.ylabel("Output")
plt.grid()

plt.subplot(2, 2, 2)
plt.plot(x, tanh)
plt.title("Tanh Function")
plt.xlabel("Sepal Length")
plt.ylabel("Output")
plt.grid()

plt.subplot(2, 2, 3)
plt.plot(x, relu)
plt.title("ReLU Function")
plt.xlabel("Sepal Length")
plt.ylabel("Output")
plt.grid()

plt.subplot(2, 2, 4)
plt.plot(x, leaky_relu)
plt.title("Leaky ReLU Function")
plt.xlabel("Sepal Length")
plt.ylabel("Output")
plt.grid()

plt.tight_layout()
plt.show()
