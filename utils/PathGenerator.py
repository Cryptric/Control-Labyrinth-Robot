from functools import partial
from multiprocessing import Pipe, Event, Process

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.backend_bases import MouseButton
from scipy.interpolate import interp1d

import Davis346Reader
from Params import *
from utils.ControlUtils import Timer, print_timers, calc_corrected_pos
from utils.FrameUtils import *


def main():
	consumer_conn, producer_conn = Pipe(False)
	termination_event = Event()
	p = Process(target=Davis346Reader.run, args=(producer_conn, termination_event))
	p.start()
	producer_conn.close()

	mask = np.zeros((IMG_SIZE_Y, IMG_SIZE_X))
	mask[CORNER_MASK_MIN_Y:CORNER_MASK_MAX_Y, CORNER_MASK_MIN_X:CORNER_MASK_MAX_X] = 1

	fig, ax = plt.subplots(nrows=1)
	img = ax.imshow(np.zeros((IMG_SIZE_Y, IMG_SIZE_X)), cmap="gray", vmin=0, vmax=255)

	frame, _ = consumer_conn.recv()
	corner_bl = calc_corrected_pos(P_CORNER_BL, 0, 0)
	corner_br = calc_corrected_pos(P_CORNER_BR, 0, 0)
	corner_tr = calc_corrected_pos(P_CORNER_TR, 0, 0)
	corner_tl = calc_corrected_pos(P_CORNER_TL, 0, 0)
	coordinate_transform_mat = calc_transform_mat([corner_bl, corner_br, corner_tr, corner_tl])
	print(coordinate_transform_mat)

	coord_points = [apply_transform(coordinate_transform_mat, corner_bl), apply_transform(coordinate_transform_mat, corner_br), apply_transform(coordinate_transform_mat, corner_tr), apply_transform(coordinate_transform_mat, corner_tl)]
	target_points = [CORNER_BL, CORNER_BR, CORNER_TR, CORNER_TL]
	mm2px_mat = calc_transform_mat(coord_points, np.array(target_points))
	print("mm to px transform matrix")
	print(mm2px_mat)
	corner_points_plt = ax.scatter([corner_br[0], corner_bl[0], corner_tl[0], corner_tr[0]], [corner_br[1], corner_bl[1], corner_tl[1], corner_tr[1]], label="detected board corners")

	path_x = []
	path_y = []

	path_plt,  = ax.plot(path_x, path_y, 'ro-', label="path")
	path_interpolated_plt,  = ax.plot(path_x, path_y, 'bo-', label="path")

	def interpolate(data):
		# https://stackoverflow.com/questions/52014197/how-to-interpolate-a-2d-curve-in-python
		distance = np.cumsum(np.sqrt(np.sum(np.diff(data, axis=0) ** 2, axis=1)))
		distance = np.insert(distance, 0, 0) / distance[-1]

		alpha = np.linspace(0, 1, 5 * len(path_x))
		interpolator = interp1d(distance, data, kind="quadratic", axis=0)
		return interpolator(alpha)

	def convert_to_mm(path_px):
		path_mm = np.zeros_like(path_px)
		for i in range(path_px.shape[0]):
			point_px = path_px[i]
			point_p_corrected = calc_corrected_pos(point_px, 0, 0)
			point_board_coordinates = apply_transform(coordinate_transform_mat, point_p_corrected)
			path_mm[i] = point_board_coordinates
		return path_mm

	def onclick(event):
		if event.button == MouseButton.LEFT:
			x, y = event.xdata, event.ydata
			if x in path_x and y in path_y:
				return
			path_x.append(x)
			path_y.append(y)
		else:
			path_x.pop()
			path_y.pop()

	def update(_, img_plt):
		with Timer("update"):
			frame, _ = consumer_conn.recv()

			path_plt.set_xdata(path_x)
			path_plt.set_ydata(path_y)

			if len(path_y) >= 3:
				path_interpolated = interpolate(np.stack((path_x, path_y), axis=1))
				path_interpolated_plt.set_xdata(path_interpolated[:, 0])
				path_interpolated_plt.set_ydata(path_interpolated[:, 1])

			img_plt.set_array(frame)
			return img_plt, corner_points_plt, path_plt, path_interpolated_plt

	update_func = partial(update, img_plt=img)
	anim = FuncAnimation(fig, update_func, cache_frame_data=False, interval=0, blit=True)

	fig.canvas.mpl_connect('button_press_event', onclick)

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
	print_timers()

	path = np.stack((path_x, path_y), axis=1)
	interpolated_path = interpolate(path)
	np.save("path-simple-labyrinth.npy", convert_to_mm(interpolated_path))


if __name__ == '__main__':
	main()
