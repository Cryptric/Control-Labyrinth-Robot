import time
from random import randrange
import socket
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import serial
from matplotlib.widgets import Slider

from pyaer import libcaer
from pyaer.davis import DAVIS

from Params import SERVO_MIN_PULSE_WIDTH
from utils.ControlUtils import send_control_signal


def record():
	device = DAVIS(noise_filter=True)
	device.start_data_stream()
	device.set_bias_from_json("./davis346_config.json")
	running = True
	events = []
	frame = None
	frame_time_camera = None
	frame_time = None

	arduino = serial.Serial('/dev/ttyUSB0', 115200, timeout=5)
	time.sleep(3)
	send_control_signal(arduino, -15, -15)
	time.sleep(2)

	print("start recording")
	while running:
		try:
			data = device.get_event()
			if data is not None:
				(pol_events, num_pol_event, special_events, num_special_event, frames_ts, frames, imu_events, num_imu_event) = data
				if num_pol_event != 0:
					events.append(pol_events)

				if isinstance(frames, int) or frames.shape[0] == 0:
					continue

				if frame_time is None and randrange(100) == 31:
					frame = frames[0]
					frame_time_camera = frames_ts
					frame_time = time.time()
					send_control_signal(arduino, 15, 15)
					if num_pol_event != 0:
						events.append(pol_events)
				elif frame_time is not None and (time.time() - frame_time) > 2:
					break
		except KeyboardInterrupt:
			break
	device.shutdown()
	return events, frame, frame_time_camera[0]


def process_events(events: List[np.ndarray], cutoff=20):
	events = np.vstack(events[cutoff:])
	events = events[events[:, 4] == 1]
	min_t = np.min(events[:, 0])
	max_t = np.max(events[:, 0])
	events_positive = events[events[:, 3] == 1]
	events_negative = events[events[:, 3] == 0]
	return events_positive, events_negative, min_t, max_t


def plot_events(events_positive, events_negative, min_t, max_t, frame, frame_ts, timeframe=10000):
	fig, ax = plt.subplots(nrows=3, height_ratios=[20, 1, 1])

	ax[0].set_xlim(0, 346)
	ax[0].set_ylim(0, 260)

	print(frame.shape)
	print(frame)
	ax[0].imshow(frame, cmap='gray')
	events_positive_plt, = ax[0].plot([0], [0], 'bo')
	events_negative_plt, = ax[0].plot([0], [0], 'ro')
	time_slider = Slider(ax[1], "Time", (min_t - frame_ts) / 1000, (max_t - frame_ts) / 1000)
	timeframe_slider = Slider(ax[2], "Timeframe", 1, 100 * 1000, valinit=timeframe)

	def plot(_):
		current_timeframe_half = timeframe_slider.val / 2
		t = time_slider.val * 1000 + frame_ts
		data_positive = events_positive[(events_positive[:, 0] > (t - current_timeframe_half/2)) & (events_positive[:, 0] < (t + current_timeframe_half/2))]
		data_negative = events_negative[(events_negative[:, 0] > (t - current_timeframe_half/2)) & (events_negative[:, 0] < (t + current_timeframe_half/2))]

		print(f"At t={time_slider.val} {data_positive.shape[0] + data_negative.shape[0]} points")

		events_positive_plt.set_xdata(data_positive[:, 1])
		events_positive_plt.set_ydata(data_positive[:, 2])

		events_negative_plt.set_xdata(data_negative[:, 1])
		events_negative_plt.set_ydata(data_negative[:, 2])

	plot(None)
	time_slider.on_changed(plot)

	plt.show()


def main():
	events, frame, frame_time_camera = record()
	events_positive, events_negative, min_t, max_t = process_events(events)
	plot_events(events_positive, events_negative, min_t, max_t, frame, frame_time_camera)


if __name__ == '__main__':
	main()
