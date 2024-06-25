import numpy as np

from HighLevelController.HighLevelController import HighLevelController
from Params import N
from utils.ControlUtils import *


class NextNearestPointController(HighLevelController):
	"""
	a path controller that should help to keep momentum
	"""
	def __init__(self):
		self.max_future_index = 5
		self.max_freeze_iterations = 20

		path = gen_path_custom_labyrinth2()
		self.path = np.zeros((path.shape[0] + N + self.max_future_index, 2))
		self.path[0:path.shape[0], :] = path
		self.path[path.shape[0]:, :] = path[-1, :]
		self.index = 1
		self.freeze_iterations = 0
		self.signal_multiplier = 1

	def get_reference_trajectory(self, pos_x, pos_y):
		if self.index + 2 * self.max_future_index + N >= self.path.shape[0]:
			self.index = 1
			return self.path[0:N]
		else:
			pos = np.array([pos_x, pos_y])
			next_index = -1
			min_distance = np.inf
			nearest = 0
			for i in range(self.index, self.index + self.max_future_index):
				ball_pos_trajectory_projection_factor = calc_projection_factor(self.path[i-1], self.path[i], pos)
				if 0 <= ball_pos_trajectory_projection_factor < 0.5:
					next_index = i
				elif 0.5 <= ball_pos_trajectory_projection_factor < 1:
					next_index = i + 1
				distance = calc_distance(pos, self.path[i])
				if distance < min_distance:
					min_distance = distance
					nearest = i
			if next_index == -1:
				print("no projection found, fall back to nearest point")
				next_index = nearest
			print(f"next point at {next_index}, moved {next_index - self.index} points")

			if next_index == self.index:
				self.freeze_iterations += 1
				if self.freeze_iterations >= self.max_freeze_iterations:
					self.signal_multiplier = min(self.signal_multiplier + 0.1, 2)
			else:
				self.freeze_iterations = 0
				self.signal_multiplier = 1
			self.index = next_index
			return self.path[self.index:self.index + N]

	def get_signal_multiplier(self):
		return self.signal_multiplier
