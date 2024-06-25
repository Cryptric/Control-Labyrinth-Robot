import numpy as np

from HighLevelController.HighLevelController import HighLevelController
from Params import N
from utils.ControlUtils import gen_path, calc_distance


class NextNearestPointController(HighLevelController):
	def __init__(self):
		self.max_future_index = 5
		self.max_freeze_iterations = 20

		path = gen_path()
		self.path = np.zeros((path.shape[0] + N + self.max_future_index, 2))
		self.path[0:path.shape[0], :] = path
		self.path[path.shape[0]:, :] = path[-1, :]
		self.index = 0
		self.freeze_iterations = 0
		self.signal_multiplier = 1

	def get_reference_trajectory(self, pos_x, pos_y):
		if self.index + 2 * self.max_future_index + N >= self.path.shape[0]:
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
			print(f"nearest point at {nearest}, moved {nearest - self.index} points, distance {distance_nearest}")
			if nearest == self.index:
				self.freeze_iterations += 1
				if self.freeze_iterations >= self.max_freeze_iterations:
					self.signal_multiplier = min(self.signal_multiplier + 0.1, 2)
			else:
				self.freeze_iterations = 0
				self.signal_multiplier = 1
			self.index = nearest
			return self.path[self.index:self.index + N]

	def get_signal_multiplier(self):
		return self.signal_multiplier
