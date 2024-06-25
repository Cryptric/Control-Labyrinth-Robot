import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from pyaer.davis import DAVIS


def record():
	device = DAVIS(noise_filter=True)
	device.start_data_stream()
	device.set_bias_from_json("./davis346_config.json")
	running = True
	events = []
	print("start recording")
	while running:
		try:
			data = device.get_event()
			if data is not None:
				(pol_events, num_pol_event, special_events, num_special_event, frames_ts, frames, imu_events, num_imu_event) = data
				if num_pol_event != 0:
					events.append(pol_events)
		except KeyboardInterrupt:
			break
	device.shutdown()
	return events


def process_events(events: List[np.ndarray], cutoff=20):
	events = np.vstack(events[cutoff:])
	events = events[events[:, 4] == 1]
	min_t = np.min(events[:, 0])
	max_t = np.max(events[:, 0])
	print(f"Recorded {events.shape[0]} events")
	events_positive = events[events[:, 3] == 1]
	events_negative = events[events[:, 3] == 0]
	return events_positive, events_negative, min_t, max_t


def plot_events(events_positive, events_negative, min_t, max_t, timeframe=10000):
	fig, ax = plt.subplots(nrows=3, height_ratios=[20, 1, 1])

	ax[0].set_xlim(0, 346)
	ax[0].set_ylim(0, 260)

	events_positive_plt, = ax[0].plot([0], [0], 'bo')
	events_negative_plt, = ax[0].plot([0], [0], 'ro')
	time_slider = Slider(ax[1], "Time", min_t, max_t)
	timeframe_slider = Slider(ax[2], "Timeframe", 1, 1000 * 1000, valinit=timeframe)

	def plot(_):
		current_timeframe_half = timeframe_slider.val / 2
		t = time_slider.val
		data_positive = events_positive[(events_positive[:, 0] > (t - current_timeframe_half/2)) & (events_positive[:, 0] < (t + current_timeframe_half/2))]
		data_negative = events_negative[(events_negative[:, 0] > (t - current_timeframe_half/2)) & (events_negative[:, 0] < (t + current_timeframe_half/2))]

		print(f"At t={t} {data_positive.shape[0] + data_negative.shape[0]} points")

		events_positive_plt.set_xdata(data_positive[:, 1])
		events_positive_plt.set_ydata(data_positive[:, 2])

		events_negative_plt.set_xdata(data_negative[:, 1])
		events_negative_plt.set_ydata(data_negative[:, 2])

	plot(None)
	time_slider.on_changed(plot)

	plt.show()


def main():
	events = record()
	events_positive, events_negative, min_t, max_t = process_events(events)
	print(max_t - min_t)
	plot_events(events_positive, events_negative, min_t, max_t)


if __name__ == '__main__':
	main()
