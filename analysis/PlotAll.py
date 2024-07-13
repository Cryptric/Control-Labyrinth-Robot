import pickle

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

from Params import *
from utils.ControlUtils import *

file_name = "2024-07-02_11-33-38.777119"


def plot_all(file):
	with open(f"store/frames_{file}.pkl", "rb") as f:
		frames = pickle.load(f)
	with open(f"store/data_{file}.pkl", "rb") as f:
		data_x, data_y = pickle.load(f)

	coordinate_transform_mat, mm2px_mat = get_transform_matrices()

	states_x = np.array(data_x["state"])
	delay_compensated_states_x = data_x["delay_compensated_state"]
	target_trajectories_x = np.array(data_x["target_trajectory"])
	predicted_states_x = data_x["predicted_state"]
	mpc_signals_x = data_x["mpc_signal"]
	signal_multipliers_x = data_x["signal_multiplier"]
	disturbance_compensations_x = data_x["disturbance_compensation"]
	board_angles_x = data_x["board_angle"]

	states_y = np.array(data_y["state"])
	delay_compensated_states_y = data_y["delay_compensated_state"]
	target_trajectories_y = np.array(data_y["target_trajectory"])
	predicted_states_y = data_y["predicted_state"]
	mpc_signals_y = data_y["mpc_signal"]
	signal_multipliers_y = data_y["signal_multiplier"]
	disturbance_compensations_y = data_y["disturbance_compensation"]
	board_angles_y = data_y["board_angle"]

	pos_xy_px = sequence_apply_transform(mm2px_mat, states_x[:, 0], states_y[:, 0]).T
	target_xy_px = sequence_apply_transform(mm2px_mat, target_trajectories_x[:, 0], target_trajectories_y[:, 0]).T

	n = len(frames)
	time_ax = np.linspace(0, dt * n, n, endpoint=False)

	frame_fig, frame_ax = plt.subplots(nrows=2, height_ratios=[20, 1])
	data_fig, data_ax = plt.subplots(nrows=2)

	index_slider = Slider(frame_ax[1], "index", 0, len(frames) - 1, valinit=0, valstep=1)

	img = frame_ax[0].imshow(np.zeros((IMG_SIZE_Y, IMG_SIZE_X)), cmap="gray", vmin=0, vmax=255)
	ball_path = frame_ax[0].plot(pos_xy_px[:, 0], pos_xy_px[:, 1], label="Ball path")
	target_path = frame_ax[0].plot(target_xy_px[:, 0], target_xy_px[:, 1], label="Target path")

	targets_x = target_trajectories_x[:, 0]
	targets_y = target_trajectories_y[:, 0]
	data_ax[0].plot(time_ax, states_x[:, 0] - targets_x, label="position_x - target_x")
	data_ax[0].plot(time_ax, states_y[:, 0] - targets_y, label="position_y - target_y")
	data_ax[0].grid()


	time_indices = []
	for ax in data_ax:
		time_indices.append(ax.axvline(x=0, color="r", label="Time index"))


	def update(_):
		i = index_slider.val

		img.set_array(remove_distortion(frames[i]))

		for time_index in time_indices:
			time_index.set_xdata([time_ax[i]] * 2)

		data_fig.canvas.draw_idle()


	index_slider.on_changed(update)

	update(None)

	def handle_close(_):
		plt.close('all')

	data_fig.canvas.mpl_connect("close_event", handle_close)
	frame_fig.canvas.mpl_connect("close_event", handle_close)

	plt.show()

def main():
	plot_all(file_name)


if __name__ == "__main__":
	main()
