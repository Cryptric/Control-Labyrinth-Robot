import time

import numpy as np
import serial


def measure(arduino: serial.Serial):
	msg = "hi;"
	t = time.time()
	arduino.write(msg.encode())
	while arduino.in_waiting < 2:
		pass
	dt = time.time() - t
	msg = arduino.read(2)
	print(f"received: {msg.decode()}")
	return dt


def main():
	arduino = serial.Serial('/dev/ttyUSB0', 115200, timeout=2)
	time.sleep(2)
	arduino.flushInput()
	arduino.flush()
	arduino.flushOutput()
	n = 1
	rtts = np.zeros(n)
	for i in range(n):
		rtt = measure(arduino)
		rtts[i] = rtt * 1000
	print(f"measured rtt avg at {np.mean(rtts)}ms (min: {np.min(rtts)}ms, max: {np.max(rtts)}ms)")


if __name__ == '__main__':
	main()
