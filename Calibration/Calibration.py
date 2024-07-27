import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression


def main():

	data_x = np.genfromtxt("Calibration/DataX.csv", skip_header=1, delimiter=",")

	data_y = np.genfromtxt("Calibration/DataY.csv", skip_header=1, delimiter=",")
	data_y[:, 1:3] *= -1

	lr_x = LinearRegression(fit_intercept=True).fit(np.expand_dims(np.tile(data_x[:, 0], 2), axis=0).T, data_x[:, 1:3].reshape(42, order='F'))
	lr_y = LinearRegression(fit_intercept=True).fit(np.expand_dims(np.tile(data_y[:, 0], 2), axis=0).T, data_y[:, 1:3].reshape(42, order='F'))

	coef_x = lr_x.coef_
	coef_y = lr_y.coef_
	bias_x = lr_x.intercept_
	bias_y = lr_y.intercept_

	print(f"K_X: {coef_x[0]}")
	print(f"K_Y: {coef_y[0]}")

	fig, axes = plt.subplots(nrows=2)

	for data, coef, bias, ax in zip([data_x, data_y], [coef_x, coef_y], [bias_x, bias_y], axes):
		ax.plot(data[:, 0], data[:, 1], label=f"Measurement +10° to -10°", linestyle="-.")
		ax.plot(data[:, 0], data[:, 2], label=f"Measurement -10° to +10°", linestyle="-.")

		ax.plot(data[:, 0], data[:, 0] * coef[0] + bias, label="Fitted Line", linewidth="2")
		ax.set_ylabel("Platform angle [°]")
		ax.set_xlabel("Servo angle [°]")
		ax.grid()
		ax.legend()

	axes[0].set_title("X-Axis")
	axes[1].set_title("Y-Axis")
	axes[0].set_ylim([-2.2, 2.2])
	axes[1].set_ylim([-2.2, 2.2])

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

	import tikzplotlib
	tikzplotlib.save("PlotData/lin-servo-platform.tex")

	plt.show()


if __name__ == '__main__':
	main()
