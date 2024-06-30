import pickle
import time

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

from Params import *
from utils.ControlUtils import SignalFilter


def main():
	with open("recorded_x.pkl", "rb") as f:
		recoded_data_x = pickle.load(f)

	with open("recorded_y.pkl", "rb") as f:
		recoded_data_y = pickle.load(f)

	states_x = np.array(recoded_data_x["state"])
	signals_x = np.array(recoded_data_x["mpc_signal"])
	disturbance_compensation_x = recoded_data_x["disturbance_compensation"]
	signal_multiplier_x = recoded_data_x["signal_multiplier"]

	states_y = np.array(recoded_data_y["state"])
	signals_y = np.array(recoded_data_y["mpc_signal"])
	disturbance_compensation_y = recoded_data_y["disturbance_compensation"]
	signal_multiplier_y = recoded_data_y["signal_multiplier"]

	fig, ax = plt.subplots()

	n = states_x.shape[0]
	time_ax = np.linspace(0, dt * n, n, endpoint=False)

	ax.plot(time_ax, signals_x[:, 0], label="signal x")
	ax.plot(time_ax, signals_y[:, 0], label="signal y", linestyle="--")

	ax.plot(time_ax, disturbance_compensation_x, label="disturbance compensation x")
	ax.plot(time_ax, disturbance_compensation_y, label="disturbance compensation y", linestyle="--")

	ax.plot(time_ax, signal_multiplier_x, label="signal multiplier x")
	ax.plot(time_ax, signal_multiplier_y, label="signal multiplier y")

	ax.plot(time_ax, (signals_x[:, 0] + disturbance_compensation_x) * signal_multiplier_x, label="final control signal x")
	ax.plot(time_ax, (signals_y[:, 0] + disturbance_compensation_y) * signal_multiplier_y, label="final control signal y", linestyle="--")

	filter = SignalFilter()
	signal_filtered = [filter(s[0]) for s in signals_x]

	ax.plot(time_ax, signal_filtered, label="filtered signal x", linestyle="-.", lw=4)

	plt.grid()
	plt.legend()
	plt.show()

if __name__ == "__main__":
	main()
