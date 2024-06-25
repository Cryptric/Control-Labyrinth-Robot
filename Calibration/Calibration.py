import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression


def main():
	data_files_x = [
		"DataX3.csv",
		"DataX4.csv",
		"DataX5.csv",
		"DataX6.csv"
	]

	data_files_y = [
		"DataY3.csv",
		"DataY4.csv",
		"DataY5.csv",
		"DataY6.csv"
	]

	data_x = []
	for f in data_files_x:
		data_x.append(np.genfromtxt(f, skip_header=1, delimiter=","))

	data_y = []
	for f in data_files_y:
		data_y.append(np.genfromtxt(f, skip_header=1, delimiter=","))

	data_x_stack = np.vstack(data_x)
	data_y_stack = np.vstack(data_y)

	coef_x = LinearRegression(fit_intercept=True).fit(np.expand_dims(data_x_stack[:, 0], 1), data_x_stack[:, 1]).coef_
	coef_y = LinearRegression(fit_intercept=True).fit(np.expand_dims(data_y_stack[:, 0], 1), data_y_stack[:, 1]).coef_

	print(f"K_X: {coef_x[0]}")
	print(f"K_Y: {coef_y[0]}")

	fig, axes = plt.subplots(nrows=2)

	for data, coef, ax in zip([data_x, data_y], [coef_x, coef_y], axes):
		for d in data:
			ax.plot(d[:, 0], d[:, 1], label="Data")
		ax.plot(data[0][:, 0], data[0][:, 0] * coef[0], label="Fitted Line")
		ax.set_ylabel("Board tilt [°]")
		ax.set_xlabel("Servo angle [°]")
		ax.grid()
		ax.legend()

	axes[0].set_title("X-Axis")
	axes[1].set_title("Y-Axis")

	# backlash_x = {}
	# for servo_angle in data_x[0][:, 0]:
	# 	s = 0
	# 	for d in data_x:
	# 		tmp = d[d[:, 0] == servo_angle][:, 1]
	# 		s = abs(tmp[0] - tmp[1])
	# 	backlash_x[servo_angle] = s / 3
	# print(backlash_x)
#
	# backlash_y = {}
	# for servo_angle in data_y[0][:, 0]:
	# 	s = 0
	# 	for d in data_y:
	# 		tmp = d[d[:, 0] == servo_angle][:, 1]
	# 		s = abs(tmp[0] - tmp[1])
	# 	backlash_y[servo_angle] = s / 3
	# print(backlash_y)

	plt.show()


if __name__ == '__main__':
	main()
