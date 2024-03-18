from math import sqrt

import numpy as np

from Params import *


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


def solve_quad(a, b, c):
	r = sqrt(b ** 2 - 4 * a * c)
	x1 = (-b + r) / (2 * a)
	x2 = (-b - r) / (2 * a)
	return x1, x2


def quad_space(start, stop, num, a1=1, b1=0, a2=1, b2=0):

	fx2 = 4 * a1 * a2
	fx1 = 4 * a2 * b1 - 4 * a1 * b2
	fx0 = (b1 - b2) ** 2 - 4 * (a1 - a2) * (start - stop)
	s_1, s_2 = solve_quad(fx2, fx1, fx0)

	s = s_1 if s_1 > s_2 else s_2
	x0 = - (b1 + 2 * a2 * s - b2) / (2 * (a1 - a2))

	x = np.linspace(0, abs(s), num)
	x_p1 = x[x <= x0]
	x_p2 = x[x > x0]

	y_p1 = a1 * x_p1 ** 2 + b1 * x_p1 + start
	y_p2 = a2 * (x_p2 - s) ** 2 + b2 * (x_p2 - s) + stop

	return np.append(y_p1, y_p2)


def calc_num_move_points(distance):
	return min(N, round(0.005 * distance / dt))


def gen_reference_path(pos, target):
	num_move_points = calc_num_move_points(abs(pos - target))
	move_points = np.linspace(pos, target, num_move_points)
	stationary_points = np.ones((N - num_move_points)) * target
	return np.append(move_points, stationary_points)


def gen_circ():
	n = 300
	w_x = np.cos(np.linspace(0, 2 * np.pi, n, endpoint=False)) * 90 + 150
	w_y = np.sin(np.linspace(0, 2 * np.pi, n, endpoint=False)) * 90 + 140
	w = np.stack((w_x, w_y), axis=1)
	return w
