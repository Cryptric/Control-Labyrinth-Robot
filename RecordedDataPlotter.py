import pickle
import time

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

from MPC import MPC
import Params


def main():
	with open("recorded_x.pkl", "rb") as f:
		recoded_data_x = pickle.load(f)

	with open("recorded_y.pkl", "rb") as f:
		recoded_data_y = pickle.load(f)

	# Example data (replace this with your own data)

	# Create initial plot
	fig, ax = plt.subplots(nrows=8, height_ratios=[10, 10, 10, 10, 10, 1, 1, 1])
	plt.title('Data Visualization')
	plt.xlabel('X')
	plt.ylabel('Y')

	ax[0].set_xlim([0, 275])
	ax[0].set_ylim([0, 235])

	ax[1].set_xlim([0, 275])
	ax[1].set_ylim([0, 235])

	start_index = 0

	# Add a slider
	index_slider = Slider(ax[5], 'Index', 0, len(recoded_data_x) - 1, valinit=start_index, valstep=1)
	control_signal_penalty_slider = Slider(ax[6], 'Scale', 0.0, 200000.0, valinit=1.0)
	du_slider = Slider(ax[7], 'Du', 0.0, abs(Params.U_min) + abs(Params.U_max), valinit=Params.U_max - Params.U_min)

	# Function to update the plot based on slider value
	def update(val):
		index = int(index_slider.val)

		draw(index)
		fig.canvas.draw_idle()

	def plot_mpc(xk_x, xk_y, ref_x, ref_y):
		mpc_x = MPC(Params.K_x, signal_cost=control_signal_penalty_slider.val, du_default=du_slider.val)
		mpc_y = MPC(Params.K_y, signal_cost=control_signal_penalty_slider.val, du_default=du_slider.val)

		t = time.time()
		signal_x_rad, _, _, _ = mpc_x.get_control_signal(ref_x, xk_x)
		signal_y_rad, _, _, _ = mpc_y.get_control_signal(ref_y, xk_y)
		dt = time.time() - t
		print(f'Time taken to compute MPC: {dt * 1000} ms')

		predicted_state_x = mpc_x.get_predicted_state(xk_x, signal_x_rad)
		predicted_state_y = mpc_y.get_predicted_state(xk_y, signal_y_rad)

		ax[1].clear()
		ax[1].set_title('Recalculated State')
		ax[1].grid()

		ax[1].plot(xk_x[0], xk_y[0], 'ro')
		ax[1].quiver(xk_x[0], xk_y[0], xk_x[1] / 10, xk_y[1] / 10, angles='xy', scale_units='xy', scale=1, color='blue')

		ax[1].plot(ref_x, ref_y, marker=".", label="ref trajectory")
		ax[1].plot(predicted_state_x, predicted_state_y, marker=".", label="prod trajectory")

		ax[2].clear()
		ax[2].set_title("New Control signal")
		ax[2].grid()

		ax[2].set_ylim(1.2 * Params.U_min, 1.2 * Params.U_max)

		ax[2].plot(np.arange(len(signal_x_rad)), signal_x_rad, 'b', label='signal x')
		ax[2].plot(np.arange(len(signal_y_rad)), signal_y_rad, 'r', label='signal y')

	def draw(index):
		state_x, _, ref_x, pred_x, signal_x, _ = recoded_data_x[index]
		state_y, _, ref_y, pred_y, signal_y, _ = recoded_data_y[index]

		ax[0].clear()
		ax[0].set_title("Recorded data")
		ax[0].grid()

		ax[0].plot(state_x[0], state_y[0], 'ro', 'Ball position')
		ax[0].quiver(state_x[0], state_y[0], state_x[1] / 10, state_y[1] / 10, angles='xy', scale_units='xy', scale=1, color='blue', label="Ball velocity / 10")

		ax[0].plot(ref_x, ref_y, marker=".", label="ref trajectory")
		ax[0].plot(pred_x, pred_y, marker=".", label="prod trajectory")

		plot_mpc(state_x, state_y, ref_x, ref_y)

		ax[0].set_xlim([0, 275])
		ax[0].set_ylim([0, 235])

		ax[1].set_xlim([0, 275])
		ax[1].set_ylim([0, 235])

		ax[3].clear()
		ax[3].set_title("Control signal")
		ax[3].grid()


		ax[3].plot(np.arange(len(signal_x)), signal_x, 'b', label='signal x')
		ax[3].plot(np.arange(len(signal_y)), signal_y, 'r', label='signal y')

		signal_x = []
		signal_y = []
		for i in range(index, min(index + Params.N, len(recoded_data_x))):
			_, _, _, _, signal_i_x, _ = recoded_data_x[i]
			_, _, _, _, signal_i_y, _ = recoded_data_y[i]
			signal_x.append(signal_i_x[0])
			signal_y.append(signal_i_y[0])

		ax[4].clear()
		ax[4].set_title("Control signal applied for next n iterations")
		ax[4].grid()

		ax[4].plot(np.arange(len(signal_x)), signal_x, 'b', label='signal x')
		ax[4].plot(np.arange(len(signal_y)), signal_y, 'r', label='signal y')

		ax[0].legend()
		ax[1].legend()
		ax[2].legend()


# Register the update function with the slider
	index_slider.on_changed(update)
	control_signal_penalty_slider.on_changed(update)
	du_slider.on_changed(update)

	draw(start_index)

	ax[0].legend()
	ax[1].legend()
	ax[2].legend()

	plt.show()


def y_state_update(pos, speed):
	pos = pos + Params.dt * speed
	speed = speed + Params.dt * Params.K_y * Params.g * (-0.07) * 5 / 7
	return pos, speed


if __name__ == '__main__':
	main()
