import numpy as np
from qpsolvers import solve_qp

from Params import *

np.set_printoptions(threshold=np.inf, linewidth=np.inf)


class MPC:
	def __init__(self):
		self.N = N
		self.dt = 0.015
		self.x0 = np.zeros((3, 1))
		self.A = np.array([[1, self.dt, 0], [0, 1, 5/7 * G * K2 * self.dt], [0, 0, 1 - K1 * self.dt]])
		self.B = np.array([[0], [0], [K1 * self.dt]])
		self.C = np.array([[1, 0, 0]])
		self.Q = Q * np.identity(N + 1)
		self.R = R * np.identity(N)
		self.w = np.zeros((self.N + 1, 1))

		self.V0 = np.zeros((N + 1, 3))
		tmp = self.C
		for i in range(N + 1):
			self.V0[i] = tmp
			tmp = tmp @ self.A
		self.S0 = np.zeros((N + 1, N))
		tmp = self.V0 @ self.B
		for i in range(N):
			self.S0[i + 1:, i] = tmp[0:N - i, 0]

		self.H = self.S0.T @ self.Q @ self.S0 + self.R
		self.g = self.S0.T @ self.Q @ (self.V0 @ self.x0 - self.w)

		self.lb = np.ones(N) * U_min
		self.ub = np.ones(N) * U_max
		self.b = np.zeros(N)
		self.h = np.ones(2 * N - 2) * du_max
		self.AM = np.eye(N, N) - np.eye(N, N, 1)
		self.AM[-1, -1] = 0
		self.GM = np.zeros((2 * N - 2, N))
		tmp = np.array([-1, 1, 1, -1])
		for i in range(self.GM.shape[1]):
			if i == 0:
				self.GM[0:2, 0] = tmp[2:4]
			elif i == self.GM.shape[1] - 1:
				self.GM[self.GM.shape[0] - 2:self.GM.shape[0], -1] = tmp[0:2]
			else:
				self.GM[2 * (i - 1):2*(i + 1), i] = tmp

	def get_control_signal(self, wk, xk):
		g = self.S0.T @ self.Q.T @ (self.V0 @ xk - wk)
		signal = solve_qp(self.H, g, G=self.GM, h=self.h, lb=self.lb, ub=self.ub, solver="osqp")
		return signal

	def get_predicted_state(self, xk, control_signal):
		return self.V0 @ xk + self.S0 @ control_signal


if __name__ == '__main__':
	mpc = MPC()
	w = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10])
	x = np.array([1, 1, 0])
	print(mpc.get_control_signal(w, x))
