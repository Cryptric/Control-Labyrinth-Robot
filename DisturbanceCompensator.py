from Params import *
import numpy as np


class DisturbanceCompensator:
	def __init__(self, initial_state, servo_2_angle_ratio):
		self.disturbance_integral = 0
		self.previous_state = initial_state
		self.prev_signals = [0] * (STEPS_DEAD_TIME + 1)
		self.K = servo_2_angle_ratio

	def update(self, measured_state, signal):
		if measured_state[0] == 0 and measured_state[1] == 0:
			return self.disturbance_integral

		d = self.previous_state[0] + dt * self.previous_state[1] + 1 / 2 * dt ** 2 * 5 / 7 * g * self.K * self.prev_signals.pop(0) - measured_state[0]

		if abs(d) <= 4:
			delta = - (14 * d) / (5 * dt**2 * g)
			I = self.disturbance_integral + delta * dt * DISTURBANCE_APPROXIMATION_INTEGRAL
			self.disturbance_integral = np.clip(I, -DISTURBANCE_INTEGRAL_CLIP, DISTURBANCE_INTEGRAL_CLIP)
		self.previous_state = measured_state
		self.prev_signals.append(signal)
		return self.disturbance_integral, d


if __name__ == '__main__':
	compensator = DisturbanceCompensator(np.array([0, 0]), 1)
	print(compensator.disturbance_integral)

	print(compensator.update(np.array([0, 0]), 4 / 180 * np.pi))
	print(compensator.update(np.array([0, 0]), 4 / 180 * np.pi))
	print(compensator.update(np.array([0, 0]), 4 / 180 * np.pi))
	print(compensator.update(np.array([0, 0]), 4 / 180 * np.pi))

	print(compensator.update(np.array([0.1, 5]), 0))
	print(compensator.update(np.array([0.1, 5]), 0))
	print(compensator.update(np.array([0.2, 5]), 0))
	print(compensator.update(np.array([0.5, 5]), 0))
