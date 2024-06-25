import time
from random import randrange

import matplotlib.pyplot as plt
import serial

from pyaer import libcaer
from pyaer.davis import DAVIS

from utils.ControlUtils import send_control_signal


def main():
	device = DAVIS(noise_filter=True)
	device.start_data_stream()
	device.set_bias_from_json("davis346_config.json")

	arduino = serial.Serial('/dev/ttyACM0', 115200, timeout=5)
	time.sleep(3)
	# send once to start procedure on arduino
	send_control_signal(arduino, 90, 90)
	time.sleep(2)

	while True:
		data = device.get_event()
		if data is None:
			continue

		(_, _, _, _, t, frames, _, _) = data
		if isinstance(frames, int) or frames.shape[0] == 0:
			continue

		if randrange(100) == 0:
			send_control_signal(arduino, 90, 90)

			print(frames.shape[0])
			plt.imshow(frames[0], cmap='gray')
			plt.show()


if __name__ == '__main__':
	main()
