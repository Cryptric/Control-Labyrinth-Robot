import ctypes
import time
import os

import numpy as np
from numpy.ctypeslib import ndpointer

from Params import STEPS_DEAD_TIME


class Simulation:
	def __init__(self, graphics):
		self.graphics = graphics
		cwd = os.getcwd()
		base_path = f"{cwd}/simulation/PhysicSimulation/cmake-build-debug"
		if graphics:
			self.simulation = ctypes.CDLL(f"{base_path}/libPhysicSimulationLib.so")
		else:
			self.simulation = ctypes.CDLL(f"{base_path}/libPhysicSimulationNoGraphicsLib.so")

		self.simulation_horizon = 30

		self.simulation.init()

		self.set_vector_field_function = self.simulation.setVectorField
		self.set_vector_field_function.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]
		self.set_vector_field_function.restype = None

		self.set_path_function = self.simulation.setPath
		self.set_path_function.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_size_t]

		self.sample_function = self.simulation.sampleControlSignal

		self.sample_function.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
										 ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
										 ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
										 ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
										 ctypes.c_size_t,
										 ctypes.POINTER(ctypes.c_float),
										 ctypes.POINTER(ctypes.c_float)]
		self.sample_function.restype = None

		if graphics:
			self.draw_function = self.simulation.drawSim
			self.draw_function.argtypes = []
			self.draw_function.restype = ctypes.c_bool

			self.close_function = self.simulation.closeSimulationPlayback
			self.close_function.argtypes = []
			self.close_function.restype = None

	def set_vector_field(self, dx, dy):
		self.set_vector_field_function(dx.astype(np.float32), dy.astype(np.float32))

	def sample_signal(self, pos, velocity, angle, prev_signals):
		signal = np.array([0, 0], dtype=np.float32)
		predicted_trajectory = np.zeros(2 * self.simulation_horizon, dtype=np.float32)
		self.sample_function(pos.astype(np.float32), velocity.astype(np.float32), angle.astype(np.float32), prev_signals.astype(np.float32), STEPS_DEAD_TIME, signal.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), predicted_trajectory.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
		return signal, predicted_trajectory.reshape((self.simulation_horizon, 2))

	def draw(self):
		if self.graphics:
			self.draw_function()

	def close(self):
		if self.graphics:
			self.close_function()


if __name__ == '__main__':
	s = Simulation(True)
	t = time.time()
	res = s.sample_signal(np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), np.array([1, 2, 3, 4]))
	dt = time.time() - t
	print(f"signal: {res}, in {dt * 1000}ms")
