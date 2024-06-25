import os
import pickle
import re

import numpy as np
from pyaer import libcaer
from pyaer.davis import DAVIS

from PIL import Image

from utils.ControlUtils import Timer, print_timers


def store_data(event_data, frame_data, path):
	"""
	stores data in specified directory, uses event-xxx.npy, frame-xxx.jpg
	:param event_data:
	:param frame_data:
	:param path:
	:return:
	"""

	print("saving data")
	p = re.compile(r"(frame|event)-\d*\.(npy|jpg)")
	files = [f for f in os.listdir(path) if p.match(f)]
	nums = [int(f[f.find("-")+1:f.find(".")]) for f in files]
	nums.append(0)
	index = max(nums) + 1

	for i in range(len(frame_data)):
		frame_file_name = path + f"frame-{index}.jpg"
		event_file_name = path + f"event-{index}.pkl"
		index = index + 1
		im = Image.fromarray(frame_data[i])
		im.save(frame_file_name)
		with open(event_file_name, "wb") as f:
			pickle.dump(event_data[i], f)
	print("data stored")


def record_data():
	frame_data = []
	event_data = [[]]

	device = DAVIS(noise_filter=True)
	device.start_data_stream()
	device.set_bias_from_json("./davis346_config.json")

	print("start recording")
	while True:
		try:
			data = device.get_event()
			if data is not None:
				with Timer("davis capture"):
					(pol_events, num_pol_event, special_events, num_special_event, frames_ts, frames, imu_events, num_imu_event) = data
					if num_pol_event > 0:
						event_data[-1].append(pol_events)
					if (not isinstance(frames, int)) and frames.shape[0] != 0:
						frame_data.append(frames[0])
						event_data.append([])
		except KeyboardInterrupt:
			print("stop recording")
			device.shutdown()
			break
	return event_data, frame_data


def main():
	path = "MLData/"
	event_data, frame_data = record_data()
	store_data(event_data, frame_data, path)

	print_timers()


if __name__ == '__main__':
	main()
