import pickle

import matplotlib.pyplot as plt
import numpy as np

from Params import *

plt.rcParams.update({'font.size': 24})


def calc_acceleration(recorded_data, cutoff=10):
	states = np.array(recorded_data["state"])
	n = states.shape[0] - cutoff
	accelerations = np.zeros(n - 1)
	prev_velocity = 0
	for i in range(n - 1):
		state = states[i + cutoff]
		velocity = state[1]
		acceleration = (velocity - prev_velocity) / dt
		accelerations[i] = acceleration
		prev_velocity = velocity
	return accelerations


def get_servo_angles(recorded_data, cutoff=10):
	mpc_signals = recorded_data["mpc_signal"]
	disturbance_compensations = recorded_data["disturbance_compensation"]
	signal_multipliers = recorded_data["signal_multiplier"]
	n = len(disturbance_compensations) - cutoff
	servo_angles = np.zeros(n)
	for i in range(n):
		signal = (mpc_signals[i + cutoff][0] + disturbance_compensations[i + cutoff]) * signal_multipliers[i + cutoff]
		servo_angles[i] = signal
	return servo_angles


def main():
	with open("recorded_x.pkl", "rb") as f:
		recoded_data_x = pickle.load(f)

	with open("recorded_y.pkl", "rb") as f:
		recoded_data_y = pickle.load(f)

	servo_angle_factor = 10000
	board_angle_facter = 100000

	acc_x, acc_y = calc_acceleration(recoded_data_x), calc_acceleration(recoded_data_y)
	servo_angles_x, servo_angles_y = get_servo_angles(recoded_data_x), get_servo_angles(recoded_data_y)
	board_angle_x, board_angle_y = np.array(recoded_data_x["board_angle"]), np.array(recoded_data_y["board_angle"])

	n = acc_x.shape[0]

	fig, axs = plt.subplots()

	x_axis = np.arange(n)
	axs.plot(x_axis, acc_x, 'b', label="Acceleration X")
	# axs.plot(x_axis, acc_y, 'r', label="Acceleration Y")

	axs.plot(x_axis, servo_angles_x[:-1] * servo_angle_factor, 'b-.', label="Servo angle X")
	# axs.plot(x_axis, servo_angles_y[:-1] * servo_angle_factor, 'r-.', label="Servo angle Y")

	axs.plot(x_axis, board_angle_x[10:-1] * board_angle_facter, 'b:', label="Approximated Board Angle X")
	# axs.plot(x_axis, board_angle_y[10:-1] * board_angle_factor, 'r:', label="Approximated Board Angle Y")

	plt.legend()
	plt.grid()
	plt.show()


if __name__ == '__main__':
	main()
