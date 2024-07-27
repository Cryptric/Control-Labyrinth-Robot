import time

import serial


def main():
	arduino = serial.Serial('/dev/ttyUSB1', 115200, timeout=5)
	time.sleep(3)

	signal = "{},{};".format(1300, 1400)
	arduino.write(bytes(signal, 'utf-8'))
	arduino.read(9)
	time.sleep(2)
	print("start measuring procedure")

	t = time.time()
	n = 1000
	for i in range(n):
		signal = "{},{};".format(1300, 1400)
		arduino.write(bytes(signal, 'utf-8'))
		m = arduino.read(9)
	dt = time.time() - t
	dt = dt / n
	print(dt * 1000)


if __name__ == '__main__':
	main()
