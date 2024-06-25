import time

import serial

from utils.ControlUtils import send_control_signal


def main():
	arduino = serial.Serial('/dev/ttyUSB0', 115200, timeout=5)
	time.sleep(5)

	# send_control_signal(arduino, 1300, 1300)
	time.sleep(2)

	t = time.time()
	send_control_signal(arduino, 1300, 1300)
	m = arduino.read(10)
	dt = time.time() - t
	print((dt) * 1000)
	print(m)


if __name__ == '__main__':
	main()