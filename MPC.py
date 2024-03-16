import numpy as np
from qpsolvers import solve_qp
from scipy import sparse

from Params import *

np.set_printoptions(threshold=np.inf, linewidth=np.inf)


class MPC:
	def __init__(self, K):
		self.N = N
		self.A = np.array([[1, dt], [0, 1]])
		self.B = np.array([[0], [dt * 5/7 * g * K]])
		self.C = np.array([[1, 0]])
		self.Q = Q * np.identity(N)
		self.R = R * np.identity(N)

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
	mpc = MPC(K_x)
	w = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10])
	x = np.array([1, 1, 0])
	print(mpc.get_control_signal(w, x))
