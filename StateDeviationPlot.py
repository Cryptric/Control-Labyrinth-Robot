import pickle
import time

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.widgets import Slider
import numpy as np
from scipy.ndimage import gaussian_filter
from Params import *
plt.rcParams.update({'font.size': 24})


# angle/unit vector function from stackoverflow https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
def unit_vector(vector):
	""" Returns the unit vector of the vector.  """
	return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
	""" Returns the angle in radians between vectors 'v1' and 'v2'  """
	v1_u = unit_vector(v1)
	v2_u = unit_vector(v2)
	return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def align_data(data_x, data_y):
	ref_pred = np.array([0, 1])

	n = len(data_x) - 1

	ref_points = []
	actual_points = []
	outliers = 0

	for i in range(n):
		state_x, state_delay_compensated_x, ref_x, pred_x, _ = data_x[i]
		state_y, state_delay_compensated_y, ref_y, pred_y, _ = data_y[i]

		state_x_ppi = data_x[i + 1][0]
		state_y_ppi = data_y[i + 1][0]

		pos = np.array([state_x[0], state_y[0]])
		pred_pos = np.array([pred_x[0], pred_y[0]])
		actual_pos = np.array([state_x_ppi[0], state_y_ppi[0]])

		pred_pos = pred_pos - pos
		actual_pos = actual_pos - pos

		pred_pos_l2 = np.linalg.norm(pred_pos)

		pred_pos = pred_pos / pred_pos_l2
		actual_pos = actual_pos / pred_pos_l2

		angle = angle_between(ref_pred, pred_pos)
		angle = - angle
		R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

		pred_pos = R @ pred_pos
		actual_pos = R @ actual_pos

		if np.linalg.norm(pred_pos - ref_pred) > 0.1:
			pred_pos = R.T @ pred_pos
			actual_pos = R.T @ actual_pos

			angle = - angle
			R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

			pred_pos = R @ pred_pos
			actual_pos = R @ actual_pos

		if np.linalg.norm(pred_pos - actual_pos) <= 2:
			ref_points.append(pred_pos)
			actual_points.append(actual_pos)
		else:
			outliers += 1

	return np.array(ref_points), np.array(actual_points), ref_pred, n, outliers


def axis_deviation(data):
	n = len(data) - 1
	deviations = []
	outliers = 0

	for i in range(n):
		state, state_delay_compensated, ref, pred, _ = data[i]

		state_ppi = data[i + 1][0]
		value = pred[0] - state_ppi[0]
		if abs(value) > 10:
			outliers += 1
		else:
			deviations.append(value)

	return np.array(deviations)


def statistics_xy(actual_points):
	mean_xy = np.mean(actual_points, axis=0)
	std_xy = np.std(actual_points, axis=0)
	cov_xy = np.cov(actual_points.T)
	return mean_xy, std_xy, cov_xy


def statistics(data):
	mean = np.mean(data, axis=0)
	std = np.std(data, axis=0)
	return mean, std


def delay_compensated_vs_actual_state(recorded_data_x, recorded_data_y):
	outliers_delay_compensation_plot = 0

	actual_positions = []
	n = len(recorded_data_x) - STEPS_DEAD_TIME
	for i in range(n):
		_, state_delay_compensated_x, _, _, _ = recorded_data_x[i]
		_, state_delay_compensated_y, _, _, _ = recorded_data_y[i]

		state_x_fi = recorded_data_x[i + STEPS_DEAD_TIME][0]
		state_y_fi = recorded_data_y[i + STEPS_DEAD_TIME][0]

		delay_compensated_position = np.array([state_delay_compensated_x[0], state_delay_compensated_y[0]])
		actual_position = np.array([state_x_fi[0], state_y_fi[0]])
		difference = actual_position - delay_compensated_position
		if np.linalg.norm(difference) <= 5:
			actual_positions.append(difference)
		else:
			outliers_delay_compensation_plot += 1

	actual_positions = np.array(actual_positions)
	return actual_positions, outliers_delay_compensation_plot


def main():
	# with open("recorded_x.pkl", "rb") as f:
	# 	recorded_data_x = pickle.load(f)
#
	# with open("recorded_y.pkl", "rb") as f:
	# 	recorded_data_y = pickle.load(f)

	with open("store/data_2024-06-02_11-20-45.906767.pkl", "rb") as f:
		recorded_data_x, recorded_data_y = pickle.load(f)


	ref_points, actual_points, ref_pred, n, outliers = align_data(recorded_data_x, recorded_data_y)

	fig, axs = plt.subplots(ncols=4)

	def heatmap(ax):
		heatmap, xedges, yedges = np.histogram2d(actual_points[:, 0], actual_points[:, 1], bins=500, range=[[-1, 1], [-0.5, 1.5]])
		heatmap = gaussian_filter(heatmap, sigma=3)
		extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

		ax.imshow(heatmap.T, extent=extent, origin='lower', cmap='inferno')
		ax.scatter(ref_pred[0], ref_pred[1], color="blue", label="normalized predicted position", s=200)
		ax.scatter([0], [0], color="red", label="current position", s=200)
		ax.quiver(0, 0, 0, 1, scale=1, scale_units='xy', color="m", label="predicted motion")

		ax.set_title("Predicted movement vs actual movement")
		ax.grid()
		ax.legend(loc='lower right')
		ax.set_ylim(-0.5, 1.5)
		ax.set_xlim(-1, 1)
		ax.set_xlabel('Normalized distance')
		ax.set_ylabel('Normalized distance')

	def plot_axis_deviation_histogram(ax, data, axis_name):
		deviation = axis_deviation(data)
		hist, bin_edges = np.histogram(deviation, bins=50)
		mean, std = statistics(deviation)

		ax.bar(bin_edges[:-1], hist, align='center')
		ax.axvline(mean, color='k', linestyle='--', label='mean')
		ax.axvline(mean - std, color='orange', linestyle='-.', label='std')
		ax.axvline(mean + std, color='orange', linestyle='-.')
		ax.set_title(f"Predicted movement vs actual movement {axis_name}")
		ax.set_xlabel("[mm]")
		ax.set_ylabel("Datapoints")
		ax.grid()
		ax.legend()

	def plot_predicted_state_vs_actual_state(ax):
		ax.scatter([0], [0], color='b', label="Predicted future state", s=200)

		actual_positions, outliers_delay_compensation_plot = delay_compensated_vs_actual_state(recorded_data_x, recorded_data_y)
		mean = np.mean(actual_positions, axis=0)
		std_x = np.std(actual_positions[:, 0])
		std_y = np.std(actual_positions[:, 1])
		print(f"Delay compensated state vs actual future state mean: {mean}, std x: {std_x}, std y: {std_y}")
		pos_heatmap, xedges, yedges = np.histogram2d(actual_positions[:, 0], actual_positions[:, 1], bins=500)
		pos_heatmap = gaussian_filter(pos_heatmap, sigma=3)
		extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
		ax.imshow(pos_heatmap.T, extent=extent, origin='lower', cmap='inferno')
		ax.add_patch(Ellipse(mean, 2 * std_x, 2 * std_y, color='orange', fill=False, label='Standard deviation', linestyle='-.'))
		ax.scatter(mean[0], mean[1], label="Mean", color='r', s=100)
		ax.set_title("Predicted latency compensated state vs actual state")
		ax.set_xlabel("[mm]")
		ax.set_ylabel("[mm]")
		ax.grid()
		ax.legend()
		print(f"removed {outliers_delay_compensation_plot} from {n} datapoints for delay compensation plot")

	print(f"removed {outliers} outliers from {n} datapoints")

	heatmap(axs[0])
	plot_axis_deviation_histogram(axs[1], recorded_data_x, "X")
	plot_axis_deviation_histogram(axs[2], recorded_data_y, "Y")
	plot_predicted_state_vs_actual_state(axs[3])

	plt.show()


if __name__ == '__main__':
	main()
