import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

from Params import *
from StateDeviationPlot import delay_compensated_vs_actual_state


def moving_avg(arr, n):
	cum_sum = np.cumsum(arr)
	cum_sum[n:] = cum_sum[n:] - cum_sum[:-n]
	return cum_sum[n - 1:] / n


def main():
	with open("recorded_x.pkl", "rb") as f:
		recorded_data_x = pickle.load(f)

	with open("recorded_y.pkl", "rb") as f:
		recorded_data_y = pickle.load(f)

	actual_positions, outliers_delay_compensation_plot = delay_compensated_vs_actual_state(recorded_data_x, recorded_data_y)

	t = np.arange(1, actual_positions.shape[0] + 1)
	div = np.vstack((t, t)).T

	cumulative_avg = actual_positions.cumsum(axis=0) / div

	fig, ax = plt.subplots(nrows=3, height_ratios=[5, 5, 1])

	ax[0].plot(t, cumulative_avg[:, 0], 'b', label="Average deviation from delay compensated predicted state X")
	ax[0].plot(t, cumulative_avg[:, 1], 'r', label="Average deviation from delay compensated predicted state Y")
	ax[0].legend()
	ax[0].grid()

	windowSlider = Slider(ax[2], "Avg window size", 1, actual_positions.shape[0], valinit=5, valstep=1)

	mv_plt_x, = ax[1].plot([0], [0], 'b', label=f"Moving average ({windowSlider.val}) deviation from delay compensated predicted state X")
	mv_plt_y, = ax[1].plot([0], [0], 'r', label=f"Moving average ({windowSlider.val}) deviation from delay compensated predicted state Y")

	def plot_moving_avg(_):
		mv_avg_x = moving_avg(actual_positions[:, 0], windowSlider.val)
		mv_avg_y = moving_avg(actual_positions[:, 1], windowSlider.val)

		mv_t = np.arange(mv_avg_x.shape[0])
		mv_plt_x.set_data(mv_t, mv_avg_x)
		mv_plt_y.set_data(mv_t, mv_avg_y)

		ax[1].relim()

	ax[1].legend()
	ax[1].grid()

	plot_moving_avg(None)
	windowSlider.on_changed(plot_moving_avg)

	plt.show()


if __name__ == '__main__':
	main()
