import numpy as np
from qpsolvers import solve_qp
from scipy import sparse

from Params import *
from utils.ControlUtils import *

np.set_printoptions(threshold=np.inf, linewidth=np.inf)


class MPC:
	def __init__(self, K, signal_cost=None, du_default=None):
		self.K = K
		self.N = N
		self.A = np.array([[1, dt], [0, 1]])
		self.B = np.array([[0], [dt * 5/7 * g * K]])
		self.C = np.array([[1, 0]])
		self.Q = np.identity(N) * (- np.exp(-0.05 * np.linspace(40, N, N) + np.log(Q)) + Q)
		self.R = (signal_cost if signal_cost else R) * np.identity(N)

		self.V0 = np.zeros((N, 2))
		tmp = self.C @ self.A
		for i in range(N):
			self.V0[i] = tmp
			tmp = tmp @ self.A

		self.S0 = np.zeros((N, N))
		tmp = self.V0 @ self.B
		# Just here for completeness, does nothing, C @ B = 0
		CB = self.C @ self.B
		tmp = np.vstack((CB, tmp))
		for i in range(N):
			self.S0[i:N, i] = tmp[0:N - i, 0]

		self.P = self.S0.T @ self.Q @ self.S0 + self.R

		self.lb = np.ones(N) * U_min
		self.ub = np.ones(N) * U_max

		self.G = np.zeros((2 * N - 2, N))
		for i in range(1, N - 1, 1):
			self.G[2 * i - 2:2 * i + 2, i] = np.array([-1, 1, 1, -1])
		self.G[0:2, 0] = np.array([1, -1])
		self.G[2 * N - 4: 2 * N - 2, N-1] = np.array([-1, 1])
		self.h = np.ones(2 * N - 2) * (du_default if du_default else du_max)

		self.P_sparse = sparse.csc_matrix(self.P)
		self.G_sparse = sparse.csc_matrix(self.G)

		self.dead_time_matrices = []
		for i in range(STEPS_DEAD_TIME):
			self.dead_time_matrices.append(np.linalg.matrix_power(self.A, i) @ self.B)
		self.dead_time_matrices.append(np.linalg.matrix_power(self.A, STEPS_DEAD_TIME))

		# stores the first entry of the control signal vector for the last STEPS_DEAD_TIME iterations
		# the most recent control signal is stored at index 0
		self.control_signal_queue = [0] * STEPS_DEAD_TIME

		# stores the delay compensated state for the last STEPS_DEAD_TIME iterations, used to then compare to the real state and approximate the disturbance in the control angle of the system
		# the most recent state is stored at index 0
		self.delay_compensated_state_queue = []

		self.angle_disturbance_integral = 0
		self.angle_disturbance_e = 0

		self.prev_signal = np.zeros(N)

	def calc_q(self, xk, wk_N):
		return self.S0.T @ self.Q @ (self.V0 @ xk - wk_N)

	def update_disturbance_approximation(self, current_xk_measured, current_xk_compensated):
		angle_disturbance = 0
		if len(self.delay_compensated_state_queue) >= STEPS_DEAD_TIME:
			e = current_xk_measured[0] - self.delay_compensated_state_queue[-1][0]
			if np.abs(e) <= 10:
				P = DISTURBANCE_APPROXIMATION_PROPORTIONAL * e
				I = self.angle_disturbance_integral + e * DISTURBANCE_APPROXIMATION_INTEGRAL * dt * low_speed_boost_factor(current_xk_measured[1])
				D = DISTURBANCE_APPROXIMATION_DIFFERENTIAL * (e - self.angle_disturbance_e) / dt
				self.angle_disturbance_e = e
				self.angle_disturbance_integral = np.clip(I, -DISTURBANCE_INTEGRAL_CLIP, DISTURBANCE_INTEGRAL_CLIP)
				angle_disturbance = np.clip(P + I + D, -DISTURBANCE_INTEGRAL_CLIP, DISTURBANCE_INTEGRAL_CLIP)
				# print(f"Angle disturbance {angle_disturbance}")
			self.delay_compensated_state_queue.pop(-1)
		self.delay_compensated_state_queue.insert(0, current_xk_compensated)
		return angle_disturbance

	def get_control_signal(self, wk, xk):
		xk_delay_compensated = self.dead_time_matrices[-1] @ xk + np.sum([self.dead_time_matrices[i] * self.control_signal_queue[i] for i in range(STEPS_DEAD_TIME)], axis=0)[:, 0]
		angle_disturbance = self.update_disturbance_approximation(xk, xk_delay_compensated)

		self.prev_signal = np.roll(self.prev_signal, -1)
		self.prev_signal[-1] = self.prev_signal[-1]
		q = self.calc_q(xk_delay_compensated, wk)
		signal = solve_qp(P=self.P_sparse, G=self.G_sparse, h=self.h, q=q, lb=self.lb, ub=self.ub, solver="osqp", initvals=self.prev_signal)
		self.prev_signal = signal
		self.control_signal_queue.insert(0, signal[0])
		prev_signal = self.control_signal_queue.pop(-1)
		return signal, angle_disturbance, xk_delay_compensated, np.sign(prev_signal) != np.sign(signal[0])

	def get_predicted_state(self, xk, control_signal):
		return self.V0 @ xk + self.S0 @ control_signal


if __name__ == '__main__':
	mpc = MPC(K_x)
	w = np.array(np.ones(N) * 10)
	x = np.array([1, 1])
	print(mpc.get_control_signal(w, x))
	print()
	print(mpc.get_predicted_state(x, mpc.get_control_signal(w, x)).shape)
