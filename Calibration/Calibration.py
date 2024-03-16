import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression


def main():
	data_x = np.genfromtxt("DataX2.csv", skip_header=1, delimiter=",")
	data_y = np.genfromtxt("DataY2.csv", skip_header=1, delimiter=",")

	coef_x = LinearRegression(fit_intercept=False).fit(np.expand_dims(data_x[:, 0], 1), data_x[:, 1]).coef_
	coef_y = LinearRegression(fit_intercept=False).fit(np.expand_dims(data_y[:, 0], 1), data_y[:, 1]).coef_

	fig, axes = plt.subplots(nrows=2)

	for data, coef, ax in zip([data_x, data_y], [coef_x, coef_y], axes):
		ax.scatter(data[:, 0], data[:, 1], label="Data", marker="x", c="r")
		ax.plot(data[:, 0], data[:, 0] * coef[0], label="Fitted Line")
		ax.set_ylabel("Board tilt [°]")
		ax.set_xlabel("Servo angle [°]")
		ax.grid()
		ax.legend()

	axes[0].set_title("X-Axis")
	axes[1].set_title("Y-Axis")

	print(f"K_X: {coef_x[0]}")
	print(f"K_Y: {coef_y[0]}")

	plt.show()


if __name__ == '__main__':
	main()
