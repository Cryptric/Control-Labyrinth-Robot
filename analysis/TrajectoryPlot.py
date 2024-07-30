import pickle
import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets import Slider

from Params import *
from utils.ControlUtils import *


def main():
	file = "2024-07-27_12-48-48.128565.pkl"

	with open(f"store/frames_{file}", "rb") as f:
		frames = pickle.load(f)
	with open(f"store/data_{file}", "rb") as f:
		data_x, data_y = pickle.load(f)

	coordinate_transform_mat, mm2px_mat = get_transform_matrices()

	states_x = np.array(data_x["state"])
	delay_compensated_states_x = data_x["delay_compensated_state"]
	target_trajectories_x = np.array(data_x["target_trajectory"])
	predicted_states_x = data_x["predicted_state"]
	mpc_signals_x = data_x["mpc_signal"]
	signal_multipliers_x = data_x["signal_multiplier"]
	disturbance_compensations_x = np.array(data_x["disturbance_compensation"])
	board_angles_x = data_x["board_angle"]

	states_y = np.array(data_y["state"])
	delay_compensated_states_y = data_y["delay_compensated_state"]
	target_trajectories_y = np.array(data_y["target_trajectory"])
	predicted_states_y = data_y["predicted_state"]
	mpc_signals_y = data_y["mpc_signal"]
	signal_multipliers_y = data_y["signal_multiplier"]
	disturbance_compensations_y = np.array(data_y["disturbance_compensation"])
	board_angles_y = data_y["board_angle"]

	pos_xy_px = sequence_apply_transform(mm2px_mat, states_x[:, 0], states_y[:, 0]).T
	target_xy_px = sequence_apply_transform(mm2px_mat, target_trajectories_x[:, 0], target_trajectories_y[:, 0]).T

	def plot_all():
		fig, ax = plt.subplots()

		for i in range(len(predicted_states_x)):
			mpc_trajectory_x = predicted_states_x[i]
			mpc_trajectory_y = predicted_states_y[i]
			mpc_trajectory_px = sequence_apply_transform(mm2px_mat, mpc_trajectory_x, mpc_trajectory_y).T
			ax.plot(mpc_trajectory_px[:, 0], mpc_trajectory_px[:, 1], linewidth=1, color="green", alpha=0.25)

		ax.imshow(remove_distortion(frames[0]), cmap="gray", vmin=0, vmax=255)

		ax.plot(pos_xy_px[:, 0], pos_xy_px[:, 1], label="Ball path")
		ax.plot(target_xy_px[:, 0], target_xy_px[:, 1], label="Target path")

		fig.legend()


	def plot_samples():
		m = len(predicted_states_x)

		n = 5
		fig, ax = plt.subplots(ncols=n)
		for i in range(n):
			start_idx = random.randint(0, len(predicted_states_x) - N)
			target_trajectory = target_xy_px[start_idx:start_idx + N]
			ball_trajectory = pos_xy_px[start_idx:start_idx + N]
			predicted_trajectory = sequence_apply_transform(mm2px_mat, predicted_states_x[start_idx], predicted_states_y[start_idx]).T

			ax[i].plot(ball_trajectory[:, 0], ball_trajectory[:, 1], label="Ball path")
			ax[i].plot(target_trajectory[:, 0], target_trajectory[:, 1], label="Target path")
			ax[i].plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], label="Predicted trajectory")
			ax[i].scatter([ball_trajectory[0, 0]], [[ball_trajectory[0, 1]]], label="Start position")

		fig.legend()

	plot_all()
	plot_samples()

	plt.show()

if __name__ == "__main__":
	main()
