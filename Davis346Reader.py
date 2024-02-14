"""DAVIS346 test example.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from pyaer import libcaer
from pyaer.davis import DAVIS

from multiprocessing.connection import Connection



def run(producer_conn: Connection):
	device = DAVIS(noise_filter=True)

	print ("Device ID:", device.device_id)
	if device.device_is_master:
		print ("Device is master.")
	else:
		print ("Device is slave.")
	print ("Device Serial Number:", device.device_serial_number)
	print ("Device String:", device.device_string)
	print ("Device USB bus Number:", device.device_usb_bus_number)
	print ("Device USB device address:", device.device_usb_device_address)
	print ("Device size X:", device.dvs_size_X)
	print ("Device size Y:", device.dvs_size_Y)
	print ("Logic Version:", device.logic_version)
	print ("Background Activity Filter:", device.dvs_has_background_activity_filter)

	device.start_data_stream()
	# setting bias after data stream started
	device.set_bias_from_json("./davis346_config.json")

	try:
		while True:
			data = device.get_event()
			if data is not None:
				(_, _, _, _, _, frames, _, _) = data
				if frames.shape[0] != 0:
					producer_conn.send(frames[0])

	finally:
		device.shutdown()
		print("davis shutdown")
