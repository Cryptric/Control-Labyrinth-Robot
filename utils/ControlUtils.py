import numpy as np


def round8(v):
	low = int(v * 10000)
	dd = low % 10000
	re = (low - low % 10000) / 10000
	if dd < 625:
		return re
	elif dd >= 625 and dd < 1875:
		return re + 0.125
	elif dd >= 1875 and dd < 3125:
		return re + 0.25
	elif dd >= 3125 and dd < 4375:
		return re + 0.375
	elif dd >= 4375 and dd < 5625:
		return re + 0.5
	elif dd >= 5625 and dd < 6875:
		return re + 0.625
	elif dd >= 6875 and dd < 8125:
		return re + 0.75
	elif dd >= 8125 and dd < 9375:
		return re + 0.875
	else:
		return re + 1.0


def find_center(output, is_weighted=True):
	center_output = np.where(output == np.amax(output))
	x_t = center_output[1][0]
	y_t = center_output[0][0]
	xsum = 0.0
	ysum = 0.0
	msum = 0.0
	for i in range(15):
		for j in range(17):
			x = i - 8 + x_t
			y = j - 8 + y_t
			xsum += output[y][x] * x
			ysum += output[y][x] * y
			msum += output[y][x]
	xr = round8(xsum / msum)
	yr = round8(ysum / msum)
	v = output[y_t][x_t]
	if is_weighted:
		return xr, yr
	mind = 1.0
	xd = 0
	yd = 0
	for i in [-1, 0, 1]:
		for j in [-1, 0, 1]:
			if i == 0 and j == 0:
				continue
			if v - output[y_t + j][x_t + i] < mind:
				mind = v - output[y_t + j][x_t + i]
				xd = i
				yd = j
	if mind < 0.008:
		x_t = x_t + 0.5 * xd
		y_t = y_t + 0.5 * yd
	return x_t, y_t


def send_control_signal(arduino, angle_x, angle_y):
	arduino.write(bytes("{},{};".format(angle_x, angle_y), 'utf-8'))


def calc_speed(pos, prev_pos, dt):
	return (pos - prev_pos) / dt
