from multiprocessing import Event, Pipe, Process

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

import Davis346Reader
from Params import *
from utils.ControlUtils import *
from utils.FrameUtils import *


def plt_with_image():
	corner_bl = calc_corrected_pos(P_CORNER_BL, 0, 0)
	corner_br = calc_corrected_pos(P_CORNER_BR, 0, 0)
	corner_tr = calc_corrected_pos(P_CORNER_TR, 0, 0)
	corner_tl = calc_corrected_pos(P_CORNER_TL, 0, 0)
	coordinate_transform_mat = calc_transform_mat([corner_bl, corner_br, corner_tr, corner_tl])

	coord_points = [apply_transform(coordinate_transform_mat, corner_bl), apply_transform(coordinate_transform_mat, corner_br), apply_transform(coordinate_transform_mat, corner_tr), apply_transform(coordinate_transform_mat, corner_tl)]
	target_points = [CORNER_BL, CORNER_BR, CORNER_TR, CORNER_TL]
	mm2px_mat = calc_transform_mat(coord_points, np.array(target_points))

	consumer_conn, producer_conn = Pipe(False)
	termination_event = Event()
	p = Process(target=Davis346Reader.run, args=(producer_conn, termination_event))
	p.start()
	producer_conn.close()

	mask = np.zeros((IMG_SIZE_Y, IMG_SIZE_X))
	mask[CORNER_MASK_MIN_Y:CORNER_MASK_MAX_Y, CORNER_MASK_MIN_X:CORNER_MASK_MAX_X] = 1

	fig, ax = plt.subplots()
	ax.grid()

	img_plt = ax.imshow(np.zeros((IMG_SIZE_Y, IMG_SIZE_X)), cmap="gray", vmin=0, vmax=255)
	corner_points_plt = ax.scatter([corner_br[0], corner_bl[0], corner_tl[0], corner_tr[0]], [corner_br[1], corner_bl[1], corner_tl[1], corner_tr[1]], label="detected board corners")

	w_circ = gen_circ()
	w_lemniscate = gen_bernoulli_lemniscate()
	w_star = gen_star()
	w_path = gen_path()
	w_path_labyrinth = gen_path_labyrinth()
	w_path_custom_labyrinth1 = gen_path_custom_labyrinth1()
	w_path_custom_labyrinth2 = gen_path_custom_labyrinth2()
	w_path_simple_labyrinth = gen_path_simple_labyrinth()
	w_path_medium_labyrinth = gen_path_medium_labyrinth()

	w_circ_px = np.array([apply_transform(mm2px_mat, w_circ[i]) for i in range(w_circ.shape[0])])
	w_lemniscate_px = np.array([apply_transform(mm2px_mat, w_lemniscate[i]) for i in range(w_lemniscate.shape[0])])
	w_star_px = np.array([apply_transform(mm2px_mat, w_star[i]) for i in range(w_star.shape[0])])
	w_path_px = np.array([apply_transform(mm2px_mat, w_path[i]) for i in range(w_path.shape[0])])
	w_path_labyrinth_px = np.array([apply_transform(mm2px_mat, w_path_labyrinth[i]) for i in range(w_path_labyrinth.shape[0])])
	w_path_custom_labyrinth1_px = np.array([apply_transform(mm2px_mat, w_path_custom_labyrinth1[i]) for i in range(w_path_custom_labyrinth1.shape[0])])
	w_path_custom_labyrinth2_px = np.array([apply_transform(mm2px_mat, w_path_custom_labyrinth2[i]) for i in range(w_path_custom_labyrinth2.shape[0])])
	w_path_simple_labyrinth_px = sequence_apply_transform(mm2px_mat, w_path_simple_labyrinth[:, 0], w_path_simple_labyrinth[:, 1])
	w_path_medium_labyrinth_px = sequence_apply_transform(mm2px_mat, w_path_medium_labyrinth[:, 0], w_path_medium_labyrinth[:, 1])

	circ_plt, = ax.plot(w_circ_px[:, 0], w_circ_px[:, 1], marker='o')
	lemniscate_plt, = ax.plot(w_lemniscate_px[:, 0], w_lemniscate_px[:, 1], marker='o')
	star_plt, = ax.plot(w_star_px[:, 0], w_star_px[:, 1], marker='o')
	path_plt, = ax.plot(w_path_px[:, 0], w_path_px[:, 1], marker='o')
	path_labyrinth_plt, = ax.plot(w_path_labyrinth_px[:, 0], w_path_labyrinth_px[:, 1], marker='x')
	path_custom_labyrinth1_plt, = ax.plot(w_path_custom_labyrinth1_px[:, 0], w_path_custom_labyrinth1_px[:, 1], marker='x')
	path_custom_labyrinth2_plt, = ax.plot(w_path_custom_labyrinth2_px[:, 0], w_path_custom_labyrinth2_px[:, 1], marker='x')
	path_simple_labyrinth_plt, = ax.plot(w_path_simple_labyrinth_px[0], w_path_simple_labyrinth_px[1], marker='x')
	path_medium_labyrinth_plt, = ax.plot(w_path_medium_labyrinth_px[0], w_path_medium_labyrinth_px[1], marker='x')

	def update(_):
		frame, _ = consumer_conn.recv()
		frame = remove_distortion(frame)
		img_plt.set_array(frame)
		return img_plt, circ_plt, lemniscate_plt, star_plt, path_plt, path_labyrinth_plt, path_custom_labyrinth1_plt, corner_points_plt, path_custom_labyrinth2_plt, path_simple_labyrinth_plt, path_medium_labyrinth_plt

	anim = FuncAnimation(fig, update, cache_frame_data=False, interval=0)
	plt.show()
	termination_event.set()
	while p.is_alive():
		try:
			if consumer_conn.poll():
				consumer_conn.recv()
		except EOFError:
			break
	consumer_conn.close()
	p.join()


def plt_without_image():
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


def main():
	with_image = True

	if with_image:
		plt_with_image()
	else:
		plt_without_image()


if __name__ == '__main__':
	main()
