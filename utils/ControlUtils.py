import time
from collections import deque
from math import sqrt

import cv2
import numpy as np
import scipy
from engineering_notation import EngNumber as eng
from scipy.interpolate import interp1d
from scipy.stats import multivariate_normal

from Params import *
from utils.FrameUtils import *

# TODO refactor this, utility functions should not manage/contain state
# always length 2, and index 1 hold most recent angle
prev_x_angles = [0, 0]
prev_y_angles = [0, 0]


def create_label(x_pos, y_pos, variance=10, img_width=240, img_height=180):
	x = np.arange(img_width)
	y = np.arange(img_height)
	X, Y = np.meshgrid(x, y)
	mu = np.array([x_pos, y_pos])
	sigma = np.array([[variance, 0], [0, variance]])
	pos = np.empty(X.shape + (2,))
	pos[:, :, 0] = X
	pos[:, :, 1] = Y
	rv = multivariate_normal(mu, sigma)
	pd = rv.pdf(pos)
	return pd


p_vals =   [[44,  34,  6,   0,  17,  25,  20,  32,  48],
			[39,   8,  8,  54,  27,  30,  21,  21,  38],
			[ 9,   3, 80, 173,  72,  36,  26,  26,  28],
			[15,  12, 78, 255, 248,  91,  54,  24,  24],
			[12,  13, 37,  96, 184, 247, 179,  32,  22],
			[23,  10, 10,  24,  61, 111, 103,  43,  24],
			[22,  13, 38, 150, 143,  39,  21,  10,  31],
			[40,  32, 19,  71,  76,  29,  29,   9,  38],
			[54,  46, 51,  18,   7,  23,  16,  31,  45]]
pattern = np.array(p_vals, dtype=np.uint8)
#pattern = (
#np.array([
#	[21, 8, 10, 20, 18, 10, 4, 12],
#	[12, 0, 2, 31, 37, 30, 7, 12],
#	[9, 1, 73, 180, 255, 230, 99, 13],
#	[11, 5, 27, 68, 109, 146, 111, 70],
#	[11, 2, 18, 32, 64, 79, 52, 19],
#	[10, 1, 5, 12, 37, 49, 28, 4],
#	[10, 1, 0, 0, 13, 21, 7, 0],
#	[13, 5, 3, 0, 1, 6, 0, 0]
#], dtype=np.uint8))
pattern_offset = np.array([5, 5])
def find_center4(frame):
	res = cv2.matchTemplate(frame, pattern, cv2.TM_CCOEFF)
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
	if max_val > 100_000:
		return np.array([max_loc[0], max_loc[1]]) + pattern_offset
	else:
		return np.array([0, 0])


def find_center2(pdf, template_size=13):
	center = int(template_size / 2)
	template = create_label(center, center, img_width=template_size, img_height=template_size)
	template = template.astype(np.float32)
	res = cv2.matchTemplate(pdf.astype(np.float32), template, method=cv2.TM_SQDIFF)
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
	min_x, min_y = min_loc
	return min_x + center, min_y + center


def find_center3(frame):
	pos = np.unravel_index(frame.argmax(), frame.shape)
	return pos[::-1]

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
			if x < 0 or y < 0 or y >= IMG_SIZE_Y or x >= IMG_SIZE_X:
				continue
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


def get_transform_matrices():
	corner_bl = calc_corrected_pos(P_CORNER_BL, 0, 0)
	corner_br = calc_corrected_pos(P_CORNER_BR, 0, 0)
	corner_tr = calc_corrected_pos(P_CORNER_TR, 0, 0)
	corner_tl = calc_corrected_pos(P_CORNER_TL, 0, 0)
	coordinate_transform_mat = calc_transform_mat([corner_bl, corner_br, corner_tr, corner_tl])

	mm2px_mat = calc_mm2px_mat(coordinate_transform_mat, corner_bl, corner_br, corner_tr, corner_tl)

	return coordinate_transform_mat, mm2px_mat


def approx_x_angle(delta_x):
	return (-sqrt(rx * (-2 * d**3 * s * delta_x + d**2 * rx * delta_x**2 + 2 * d * rx**2 * s * delta_x + rx**3 * s**2)) + d * rx * delta_x + rx**2 * s) / (d * rx * s)


def approx_y_angle(delta_y, alpha_x):
	return (-sqrt(ry * (np.cos(alpha_x)**2 * ry * (d * delta_y + ry * s)**2 - 2 * d**3 * s * delta_y)) + np.cos(alpha_x) * ry * (d * delta_y + ry * s))/(d * ry * s)


def calc_board_angle(frame, original_focal_pos, prev_x_angle=0, prev_y_angle=0):
	focal_x_pos = detect_focal_x(frame)
	focal_y_pos = detect_focal_y(frame)
	focal_pos = np.array([focal_x_pos[0], focal_y_pos[1]])
	focal_displacement_px = original_focal_pos - focal_pos
	focal_displacement_px = focal_displacement_px * (-1)  # camera mirrors image
	focal_displacement_px[1] *= -1  # x marker is on negative side (center coordinates), while y marker is on positive side
	focal_displacement_mm = focal_displacement_px * PIXEL_SIZE
	try:
		angle_x = approx_x_angle(focal_displacement_mm[0]) if focal_x_pos[0] != 0 and focal_x_pos[1] != 0 else prev_x_angle
		angle_y = approx_y_angle(focal_displacement_mm[1], angle_x) if focal_y_pos[0] != 0 and focal_y_pos[1] != 0 else prev_y_angle
		return angle_x, angle_y
	except:
		return prev_x_angle, prev_y_angle


def calc_corrected_pos(pos, angle_x, angle_y):
	pos = pos - np.array([IMG_SIZE_X / 2, IMG_SIZE_Y / 2])
	pos = pos * PIXEL_SIZE
	pos = pos * -1

	# beta is angle angle x
	lambda_x = -5 * pos[0] * np.sin(angle_x) * np.cos(angle_y) + 41 * np.cos(angle_x)
	lambda_y = 5 * pos[0] * np.sin(angle_y)
	lambda_b = -2050 * pos[0]

	phi_x = -5 * pos[1] * np.sin(angle_x) * np.cos(angle_y) + 41 * np.sin(angle_x) * np.sin(angle_y)
	phi_y = 41 * np.cos(angle_y) + 5 * pos[1] * np.sin(angle_y)
	phi_b = -2050 * pos[1]

	factor = 1 / (lambda_x * phi_y - lambda_y * phi_x)

	px = factor * (lambda_b * phi_y - lambda_y * phi_b)
	py = factor * (-lambda_b * phi_x + lambda_x * phi_b)

	return np.array([px, py]) + np.array([IMG_SIZE_X / 2, IMG_SIZE_Y / 2])


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

	signal = "{},{};".format(pw_x, pw_y)
	arduino.write(bytes(signal, 'utf-8'))


def get_backlash_compensation_term(prev_angles, angle, backlash_table):
	prev_dir = np.sign(prev_angles[1] - prev_angles[0])
	current_dir = np.sign(angle - prev_angles[1])
	if prev_dir != current_dir:
		return -current_dir * backlash_table[round(prev_angles[1])]
	return 0


def calc_speed(pos, prev_pos, dt):
	return (pos - prev_pos) / dt


def low_speed_boost_factor(speed):
	factor = - np.arctan(abs(speed)) / np.pi * 4 + 3
	return factor


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


def calc_distance(x, y):
	return np.linalg.norm(x - y)


def calc_projection_factor(base_point, target, vector):
	u = target - base_point
	v = vector - base_point
	return (u @ v) / (u @ u)


def interpolate(data, n=3000):
	# https://stackoverflow.com/questions/52014197/how-to-interpolate-a-2d-curve-in-python
	distance = np.cumsum(np.sqrt(np.sum(np.diff(data, axis=0) ** 2, axis=1)))
	distance = np.insert(distance, 0, 0) / distance[-1]

	alpha = np.linspace(0, 1, n)
	interpolator = interp1d(distance, data, kind="quadratic", axis=0)
	return interpolator(alpha)


def gen_reference_path(pos, target):
	num_move_points = calc_num_move_points(abs(pos - target))
	move_points = np.linspace(pos, target, num_move_points)
	stationary_points = np.ones((N - num_move_points)) * target
	return np.append(move_points, stationary_points)


def gen_circ():
	n = 450
	w_x = np.cos(np.linspace(0, 2 * np.pi, n, endpoint=False)) * 90 + 137.5
	w_y = np.sin(np.linspace(0, 2 * np.pi, n, endpoint=False)) * 90 + 115
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


def gen_path():
	path = np.load("path.npy")
	return interpolate(path)


def gen_path_labyrinth():
	path = np.load("path-labyrinth.npy")
	path[:, 0] = path[:, 0] - 3
	return interpolate(path)


def gen_path_custom_labyrinth1():
	path = np.load("path-custom-labyrinth-1.npy")
	path[:, 0] = path[:, 0] - 3
	return interpolate(path, n=1500)


def gen_path_custom_labyrinth2():
	path = np.load("path-custom-labyrinth-2.npy")
	path[:, 0] = path[:, 0] - 3
	path = interpolate(path, n=750)
	path[:, 1] = path[:, 1] - 8
	path[:, 0] = path[:, 0] - 3
	return path


def gen_path_simple_labyrinth():
	path = np.load("path-simple-labyrinth.npy")
	path[:, 0] = path[:, 0] - 3
	path[:, 1] = (path[:, 1] - BOARD_LENGTH_Y / 2) * 0.97 + BOARD_LENGTH_Y / 2
	path[:, 0] = (path[:, 0] - BOARD_LENGTH_X) * 0.97 + BOARD_LENGTH_X
	return interpolate(path, n=2000)


def gen_path_medium_labyrinth():
	path = np.load("path-medium-labyrinth.npy")
	path[:, 0] = path[:, 0] - 3
	path[:, 1] = (path[:, 1] - BOARD_LENGTH_Y / 2) * 0.97 + BOARD_LENGTH_Y / 2
	path[:, 0] = (path[:, 0] - BOARD_LENGTH_X) * 0.97 + BOARD_LENGTH_X
	return interpolate(path, n=2000)


def calc_following_mse(recorded_data):
	# ignore first 10 seconds to give the ball some time to catch up, otherwise starting point has huge influence on quality measure
	positions = np.array(recorded_data["state"])[int(10 / dt):, 0]
	next_ref_points = np.array(recorded_data["target_trajectory"])[int(10 / dt):, 0]
	errors = (positions[1:] - next_ref_points[0:-1]) ** 2
	return np.sum(errors) / (errors.shape[0])


def calc_control_signal_smoothness_measure(recorded_data):
	control_signals = np.array(recorded_data["mpc_signal"])[int(10 * 1 / dt):, 0]
	n = len(control_signals)
	measure = 0
	for i in range(n - 1):
		signal = control_signals[i]
		signal_next = control_signals[i + 1]
		measure += abs(signal - signal_next)
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


# https://www.samproell.io/posts/yarppg/yarppg-live-digital-filter/
class SignalFilter:
	def __init__(self):
		self.b, self.a = scipy.signal.iirfilter(4, Wn=15, fs=1/dt, btype="low", ftype="butter")
		self._xs = deque([0] * len(self.b), maxlen=len(self.b))
		self._ys = deque([0] * (len(self.a) - 1), maxlen=len(self.a)-1)

	def __call__(self, x):
		"""Filter incoming data with standard difference equations.
		"""
		self._xs.appendleft(x)
		y = np.dot(self.b, self._xs) - np.dot(self.a[1:], self._ys)
		y = y / self.a[0]
		self._ys.appendleft(y)
		return y
