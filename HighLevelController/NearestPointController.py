import numpy as np

from HighLevelController.HighLevelController import HighLevelController
from Params import N
from utils.ControlUtils import *


class NearestPointController(HighLevelController):
	def __init__(self, path):
		self.max_future_index = 5
		self.max_freeze_iterations = 20

		self.path = np.zeros((path.shape[0] + N + self.max_future_index, 2))
		self.path[0:path.shape[0], :] = path
		self.path[path.shape[0]:, :] = path[-1, :]
		self.index = 0
		self.freeze_iterations = 0
		self.signal_multiplier = 1

	def get_reference_trajectory(self, pos_x, pos_y):
		if self.index + 6 * self.max_future_index + N >= self.path.shape[0]:
			self.index = 0
			return self.path[0:N]
		else:
			pos = np.array([pos_x, pos_y])
			nearest = 0
			distance_nearest = np.inf
			for i in range(self.index, self.index + self.max_future_index):
				distance = calc_distance(pos, self.path[i])
				if distance < distance_nearest:
					nearest = i
					distance_nearest = distance
			if nearest == self.index:
				self.freeze_iterations += 1
				if self.freeze_iterations >= self.max_freeze_iterations and (self.freeze_iterations - self.max_freeze_iterations):
					self.signal_multiplier = min(self.signal_multiplier + 0.05, 2.5)
					# print(f"signal_multiplier: {self.signal_multiplier}")
			else:
				self.freeze_iterations = 0
				self.signal_multiplier = max(1.0, 0.99 + (self.signal_multiplier - 1) / 2)
			self.index = nearest
			return self.path[self.index + 4:self.index + N + 4]

	def get_signal_multiplier(self, deactivate):
		if deactivate:
			self.signal_multiplier = max(1.0, 0.99 + (self.signal_multiplier - 1) / 2)
		return self.signal_multiplier
