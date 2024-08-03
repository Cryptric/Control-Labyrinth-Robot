import matplotlib.pyplot as plt
import numpy as np

from DisturbanceCompensator import DisturbanceCompensator
from HighLevelController.HighLevelController import HighLevelController
from MPC import MPC
from Params import *
from ScoreMatrixGenerator import gen_neg_gradient_field
from simulation.SimulationInterface import Simulation
from utils.ControlUtils import gen_circ


class Controller:

	def __init__(self):
		self.recorded_data_x = None
		self.recorded_data_y = None

	def set_recorders(self, data_x, data_y):
		self.recorded_data_x = data_x
		self.recorded_data_y = data_y

	def __call__(self, pos, velocity):
		pass

	def destroy(self):
		pass

	def visualization_update(self):
		pass


class LinearMPC(Controller):
	def __init__(self, pos, high_level_controller: HighLevelController):
		super().__init__()
		self.path_controller = high_level_controller

		self.mpc_x = MPC(K_x)
		self.mpc_y = MPC(K_y)

		self.disturbance_compensator_x = DisturbanceCompensator(np.array([pos[0], 0]), K_x)
		self.disturbance_compensator_y = DisturbanceCompensator(np.array([pos[1], 0]), K_y)

		self.w_circ = gen_circ()

	def __call__(self, pos, velocity):
		xk_x = np.array([pos[0], velocity[0]])
		xk_y = np.array([pos[1], velocity[1]])

		#target_trajectory = self.w_circ[0:N]
		target_trajectory = self.path_controller.get_reference_trajectory(pos[0], pos[1])

		signal_x_rad, deactivate_multiplier_x = self.mpc_x.get_control_signal(target_trajectory[:, 0], xk_x)
		signal_y_rad, deactivate_multiplier_y = self.mpc_y.get_control_signal(target_trajectory[:, 1], xk_y)

		predicted_states_x = self.mpc_x.get_predicted_state(xk_x, signal_x_rad)
		predicted_states_y = self.mpc_y.get_predicted_state(xk_y, signal_y_rad)

		signal_multiplier = self.path_controller.get_signal_multiplier(deactivate_multiplier_x or deactivate_multiplier_y)

		# x_path_diff_norm = np.linalg.norm(target_trajectory[:, 0] - predicted_state_x)
		# y_path_diff_norm = np.linalg.norm(target_trajectory[:, 1] - predicted_state_y)
		signal_multiplier_x = 1
		signal_multiplier_y = 1
		# TODO if signal_multiplier != 1:
		# TODO 	l = signal_multiplier / np.sqrt(x_path_diff_norm**2 + y_path_diff_norm**2)
		# TODO 	signal_multiplier_x = max(1, l * x_path_diff_norm)
		# TODO 	signal_multiplier_y = max(1, l * y_path_diff_norm)

		disturbance_x, d_x = self.disturbance_compensator_x.update(xk_x, signal_x_rad[STEPS_DEAD_TIME])
		disturbance_y, d_y = self.disturbance_compensator_y.update(xk_y, signal_y_rad[STEPS_DEAD_TIME])

		self.recorded_data_x["delta"].append(d_x)
		self.recorded_data_y["delta"].append(d_y)

		# self.w_circ = np.roll(self.w_circ, -1, axis=0)

		return (signal_x_rad, signal_y_rad), (signal_multiplier_x, signal_multiplier_y), (disturbance_x, disturbance_y), (predicted_states_x, predicted_states_y), target_trajectory


class SimulationController(Controller):
	def __init__(self, pos):
		super().__init__()
		self.signal_queue = [np.array([0, 0], dtype=np.float32)] * STEPS_DEAD_TIME

		self.simulation = Simulation(True)
		dx, dy = gen_neg_gradient_field()
		self.simulation.set_vector_field(dx, dy)

		self.disturbance_compensator_x = DisturbanceCompensator(np.array([pos[0], 0]), K_x)
		self.disturbance_compensator_y = DisturbanceCompensator(np.array([pos[1], 0]), K_y)

		self.debug_data = []

	def __call__(self, pos, velocity):
		prev_signals = np.concatenate(self.signal_queue)
		print(prev_signals)
		self.debug_data.append(prev_signals)
		sim_sig, pred_trajectory = self.simulation.sample_signal(np.array(pos), np.array(velocity), np.array([0, 0]), prev_signals)

		self.signal_queue.pop(0)
		self.signal_queue.append(sim_sig.copy())

		# convert to servo angles
		print(sim_sig)
		sim_sig[0] = sim_sig[0] / K_x
		sim_sig[1] = sim_sig[1] / K_y

		disturbance_x, d_x = 0, 0 # self.disturbance_compensator_x.update([pos[0], velocity[0]], sim_sig[0])
		disturbance_y, d_y = 0, 0 # self.disturbance_compensator_y.update([pos[1], velocity[1]], sim_sig[1])

		self.recorded_data_x["delta"].append(d_x)
		self.recorded_data_y["delta"].append(d_y)

		return ([sim_sig[0]] * (STEPS_DEAD_TIME + 1), [sim_sig[1]] * (STEPS_DEAD_TIME + 1)), (1, 1), (disturbance_x, disturbance_y), (pred_trajectory[:, 0], pred_trajectory[:, 1]), np.array([pos])

	def visualization_update(self):
		self.simulation.draw()

	def destroy(self):
		self.simulation.close()

