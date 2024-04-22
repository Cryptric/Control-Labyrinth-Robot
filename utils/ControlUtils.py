import time
from math import sqrt

import numpy as np
from engineering_notation import EngNumber as eng

from Params import *


# TODO refactor this, utility functions should not manage/contain state
# always length 2, and index 1 hold most recent angle
prev_x_angles = [0, 0]
prev_y_angles = [0, 0]

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
	xr = (xsum / msum)
	yr = (ysum / msum)
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


def map_value_range(x, in_min, in_max, out_min, out_max):
	return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def send_control_signal(arduino, angle_x, angle_y):
	global prev_x_angles
	global prev_y_angles

	# angle_x = angle_x + get_backlash_compensation_term(prev_x_angles, angle_x, servo_backlash_x)
	# angle_y = angle_y + get_backlash_compensation_term(prev_y_angles, angle_y, servo_backlash_y)

	# prev_x_angles.append(angle_x)
	# prev_x_angles.pop(0)
	# prev_y_angles.append(angle_y)
	# prev_y_angles.pop(0)

	angle_x = angle_x + X_CONTROL_SIGNAL_HORIZONTAL
	angle_y = angle_y + Y_CONTROL_SIGNAL_HORIZONTAL

	# map servo angle to pulse width
	pw_x = map_value_range(angle_x, 0, 180, SERVO_MIN_PULSE_WIDTH, SERVO_MAX_PULSE_WIDTH)
	pw_y = map_value_range(angle_y, 0, 180, SERVO_MIN_PULSE_WIDTH, SERVO_MAX_PULSE_WIDTH)

	arduino.write(bytes("{},{};".format(pw_x, pw_y), 'utf-8'))


def get_backlash_compensation_term(prev_angles, angle, backlash_table):
	prev_dir = np.sign(prev_angles[1] - prev_angles[0])
	current_dir = np.sign(angle - prev_angles[1])
	if prev_dir != current_dir:
		return -current_dir * backlash_table[round(prev_angles[1])]
	return 0


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
	n = 450
	w_x = np.cos(np.linspace(0, 2 * np.pi, n, endpoint=False)) * 90 + 150
	w_y = np.sin(np.linspace(0, 2 * np.pi, n, endpoint=False)) * 90 + 140
	w = np.stack((w_x, w_y), axis=1)
	return w


def gen_bernoulli_lemniscate():
	n = 600
	a = 90
	t = np.linspace(0, 2 * np.pi, n, endpoint=False)
	w_x = (a * np.cos(t)) / (1 + np.sin(t) ** 2) + 150
	w_y = (a * np.sin(t) * np.cos(t)) / (1 + np.sin(t) ** 2) + 140
	w = np.stack((w_x, w_y), axis=1)
	return w


def gen_star():
	# from https://math.stackexchange.com/questions/1308602/how-to-calculate-the-intersection-points-of-the-same-implicit-curve-in-parametri
	n = 1000
	t = np.linspace(0, 2 * np.pi, n, endpoint=False)
	w_x = (27/14 * np.sin(2 * t) + 15/14 * np.sin(3 * t)) * 25 + 150
	w_y = (27/14 * np.cos(2 * t) - 15/14 * np.cos(3 * t)) * 25 + 140
	w = np.stack((w_x, w_y), axis=1)
	return w


def calc_following_mse(recorded_data):
	# ignore first 5 seconds to give the ball some time to catch up, otherwise starting point has huge influence on quality measure
	recorded_data = recorded_data[int(5 * 1 / dt):]
	n = len(recorded_data)
	error = 0
	for i in range(n - 1):
		_, ref, _, _ = recorded_data[i]
		x_next, _, _, _ = recorded_data[i + 1]
		error += (x_next[0] - ref[0]) ** 2
	return error / (n - 1)


def calc_control_signal_smoothness_measure(recorded_data):
	recorded_data = recorded_data[int(5 * 1 / dt):]
	n = len(recorded_data)
	measure = 0
	for i in range(n - 1):
		_, _, _, signal = recorded_data[i]
		_, _, _, signal_next = recorded_data[i + 1]
		measure += abs(signal[0] - signal_next[0])
	return measure / (n - 1)


timers = {}
times = {}
enter_times = {}
class Timer:
	def __init__(self, timer_name='', delay=None, show_hist=False, numpy_file=None):
		""" Make a Timer() in a _with_ statement for a block of code.
		The timer is started when the block is entered and stopped when exited.
		The Timer _must_ be used in a with statement.

		:param timer_name: the str by which this timer is repeatedly called and which it is named when summary is printed on exit
		:param delay: set this to a value to simply accumulate this externally determined interval
		:param show_hist: whether to plot a histogram with pyplot
		:param numpy_file: optional numpy file path
		"""
		self.timer_name = timer_name
		self.show_hist = show_hist
		self.numpy_file = numpy_file
		self.delay = delay

		if self.timer_name not in timers.keys():
			timers[self.timer_name] = self
		if self.timer_name not in times.keys():
			times[self.timer_name] = []
			enter_times[self.timer_name] = []

	def __enter__(self):
		if self.delay is None:
			self.start = time.time()
			enter_times[self.timer_name].append(self.start)
		return self

	def __exit__(self, *args):
		if self.delay is None:
			self.end = time.time()
			self.interval = self.end - self.start  # measured in seconds
		else:
			self.interval = self.delay
		times[self.timer_name].append(self.interval)

	# print(self.timer_name, self.interval)

	def print_timing_info(self, logger=None):
		""" Prints the timing information accumulated for this Timer

		:param logger: write to the supplied logger, otherwise use the built-in logger
		"""
		if len(times) == 0:
			print(f'Timer {self.timer_name} has no statistics; was it used without a "with" statement?')
			return
		a = np.array(times[self.timer_name])
		timing_mean = np.mean(a)  # todo use built in print method for timer
		timing_std = np.std(a)
		timing_median = np.median(a)
		timing_min = np.min(a)
		timing_max = np.max(a)
		s = '{} n={}: {}s +/- {}s (median {}s, min {}s max {}s)'.format(self.timer_name, len(a),
																		eng(timing_mean), eng(timing_std),
																		eng(timing_median), eng(timing_min),
																		eng(timing_max))
		b = np.array(enter_times[self.timer_name])
		b = 1 / (b[1:] - b[:-1])
		freq_mean = np.mean(b)
		freq_std = np.std(b)
		freq_median = np.median(b)
		freq_min = np.min(b)
		freq_max = np.max(b)
		s2 = f"call frequency: {freq_mean:.1f}Hz +/- {freq_std:.1f}Hz (median {freq_median:.1f}Hz, min {freq_min:.1f}Hz, max {freq_max:.1f}Hz)"
		s = s + s2
		if logger is not None:
			logger.info(s)
		else:
			print(s)


def print_timers():
	for _, timer in timers.items():
		timer.print_timing_info()
