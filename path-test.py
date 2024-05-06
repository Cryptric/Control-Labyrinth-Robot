import numpy as np
from matplotlib import pyplot as plt


from Params import *
from utils.ControlUtils import gen_circ, gen_bernoulli_lemniscate, gen_star, gen_path


def main():
	w_circ = gen_circ()
	w_lemniscate = gen_bernoulli_lemniscate()
	w_star = gen_star()
	w_path = gen_path()

	plt.plot(w_circ[:, 0], w_circ[:, 1], marker='o')
	plt.plot(w_lemniscate[:, 0], w_lemniscate[:, 1], marker='o')
	plt.plot(w_star[:, 0], w_star[:, 1], marker='o')
	plt.plot(w_path[:, 0], w_path[:, 1], marker='o')
	plt.grid()
	plt.gca().set_aspect('equal')
	plt.show()


if __name__ == '__main__':
	main()
