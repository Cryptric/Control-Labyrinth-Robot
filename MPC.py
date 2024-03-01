import numpy as np
from qpsolvers import solve_qp
from scipy import sparse

from Params import *

np.set_printoptions(threshold=np.inf, linewidth=np.inf)


class MPC:
	def __init__(self):
		self.N = N
		self.A = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])
		self.B = np.array([[0, 0], [dt * 5/7 * g * K_x, 0], [0, 0], [0, dt * 5/7 * g * K_y]])
		self.C = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
		self.Q = Q * np.identity(2 * N)
		self.R = R * np.identity(2 * N)

		self.V0 = np.zeros((2 * N, 4))
		tmp = self.C @ self.A
		for i in range(N):
			self.V0[2 * i:2 * (i + 1)] = tmp
			tmp = tmp @ self.A

		self.S0 = np.zeros((2 * N, 2 * N))
		tmp = self.V0 @ self.B
		# Just here for completeness, does nothing, C @ B = 0
		CB = self.C @ self.B
		for i in range(2 * N - 1):
			self.S0[i+2:2 * N, i] = tmp[0:2 * N - (i + 2), 0]
			self.S0[i:i+2, i:i+2] = CB
		self.S0[2 * N - 2:2 * N, 2 * N - 2:2 * N] = CB

		self.P = self.S0.T @ self.Q @ self.S0 + self.R

		self.lb = np.ones(2 * N) * U_min
		self.ub = np.ones(2 * N) * U_max

		self.P_sparse = sparse.csr_matrix(self.P)

	def calc_q(self, xk, wk_N):
		return self.S0.T @ self.Q @ (self.V0 @ xk - wk_N)

	def get_control_signal(self, wk, xk):
		q = self.calc_q(xk, wk)
		signal = solve_qp(P=self.P_sparse, q=q, lb=self.lb, ub=self.ub, solver="osqp")
		return signal

	def get_predicted_state(self, xk, control_signal):
		return self.V0 @ xk + self.S0 @ control_signal


if __name__ == '__main__':
	mpc = MPC()
	w = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10])
	x = np.array([1, 1, 0])
	print(mpc.get_control_signal(w, x))
