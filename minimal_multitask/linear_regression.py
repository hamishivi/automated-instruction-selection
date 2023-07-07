import argparse

import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d",
    "--data",
    type=str,
    help="Path to csv data file. feature to predict should be final col.",
    required=True,
)
parser.add_argument(
    "-t", "--test", type=float, help="Fraction of data to use for test.", default=0.2
)
args = parser.parse_args()

# Read in data.
data = np.loadtxt(args.data, delimiter=",", skiprows=1)

# shuffle data
np.random.shuffle(data)

# train-test split.
train, test = data[: int(len(data) * (1 - args.test))], data[int(len(data) * (1 - args.test)) :]

# Split into X and y.
x, y = train[:, :-1], train[:, -1]

# Fit model.
model = LinearRegression()
model.fit(x, y)

# Evaluate model.
print("Train score: ", model.score(x, y))
print("Test score: ", model.score(test[:, :-1], test[:, -1]))
print("Test loss: ", np.mean((model.predict(test[:, :-1]) - test[:, -1]) ** 2))
print("Test R^2: ", np.corrcoef(model.predict(test[:, :-1]), test[:, -1])[0, 1] ** 2)

# Print coefficients.
print("Coefficients: ", model.coef_)
print("Intercept: ", model.intercept_)

# Plot predictions.
plt.scatter(model.predict(test[:, :-1]), test[:, -1])
plt.scatter(np.mean(test[:, :-1], axis=-1), test[:, -1])
plt.plot([0, 1], [0, 1], transform=plt.gca().transAxes)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("predictions.png")
