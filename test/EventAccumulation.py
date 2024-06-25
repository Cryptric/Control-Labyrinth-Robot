"""DAVIS346 test example.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function

import time

import cv2
import numpy as np
from pyaer.davis import DAVIS
from PIL import Image

from utils.ControlUtils import print_timers

device = DAVIS(noise_filter=True)

print("Device ID:", device.device_id)
if device.device_is_master:
	print("Device is master.")
else:
	print("Device is slave.")
print("Device Serial Number:", device.device_serial_number)
print("Device String:", device.device_string)
print("Device USB bus Number:", device.device_usb_bus_number)
print("Device USB device address:", device.device_usb_device_address)
print("Device size X:", device.dvs_size_X)
print("Device size Y:", device.dvs_size_Y)
print("Logic Version:", device.logic_version)
print("Background Activity Filter:", device.dvs_has_background_activity_filter)

device.start_data_stream()
# setting bias after data stream started
device.set_bias_from_json("./davis346_config.json")

clip_value = 3
histrange = [(0, v) for v in (260, 346)]


store_path = "RecordTest"

def get_event(device):
	data = device.get_event()

	return data


num_packet_before_disable = 1000

EVENT_COUNT_PER_FRAME = 10

cv2_resized_frame = False

img_counter = 0

last_frame = np.zeros((346, 260), dtype=np.uint8)

while True:
	try:
		t = time.time()
		events = None
		while events is None or len(events) < EVENT_COUNT_PER_FRAME:
			data = device.get_event()
			(pol_events, num_pol_event, special_events, num_special_event, frames_ts, frames, imu_events, num_imu_event) = data
			if frames.shape[0] != 0:
				last_frame = frames[0]
				frame = cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)
				cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
				cv2.imshow('frame', frame)
				if not cv2_resized_frame:
					cv2.resizeWindow('frame', 600, 600)
					cv2_resized_frame = True
			if num_pol_event > 0:
				pol_events = pol_events[pol_events[:, 4] == 1]
				if events is None:
					events = pol_events
				else:
					events = np.vstack([events, pol_events])
			if cv2.waitKey(1) & 0xFF == ord('q'):
				raise KeyboardInterrupt
			# print(f"accumulated {events.shape[0] if events is not None else 0} events")

		pimg = Image.fromarray(last_frame)
		pimg.save(f"{store_path}/img-{img_counter}.jpg")
		np.save(f"{store_path}/events-{img_counter}.npy", events)
		img_counter += 1

		pol_on = (events[:, 3] == 1)
		pol_off = np.logical_not(pol_on)
		img_on, _, _ = np.histogram2d(events[pol_on, 2], events[pol_on, 1], bins=(260, 346), range=histrange)
		img_off, _, _ = np.histogram2d(events[pol_off, 2], events[pol_off, 1], bins=(260, 346), range=histrange)
		if clip_value is not None:
			integrated_img = np.clip((img_on - img_off), -clip_value, clip_value)
		else:
			integrated_img = (img_on - img_off)
		img = integrated_img + clip_value

		dt = time.time() - t
		print(f"dt: {dt * 1000}ms with {events.shape[0]} events")
		cv2.imshow("image", img / float(clip_value * 2))
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	except KeyboardInterrupt:
		print("shutdown davis")
		device.shutdown()
		break

print_timers()
