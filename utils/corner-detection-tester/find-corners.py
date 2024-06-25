from functools import partial
from multiprocessing import Pipe, Event, Process

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

import Davis346Reader
from Params import *
from utils.ControlUtils import Timer, print_timers
from utils.FrameUtils import remove_distortion, find_board_corners, find_corners, process_frame
from utils.Plotting import pr_cmap

matplotlib.use("TkAgg")


def main():
	consumer_conn, producer_conn = Pipe(False)
	termination_event = Event()
	p = Process(target=Davis346Reader.run, args=(producer_conn, termination_event))
	p.start()
	producer_conn.close()

	mask = np.zeros((IMG_SIZE_Y, IMG_SIZE_X))
	mask[CORNER_MASK_MIN_Y:CORNER_MASK_MAX_Y, CORNER_MASK_MIN_X:CORNER_MASK_MAX_X] = 1

	fig, ax = plt.subplots(nrows=5, height_ratios=[6, 1, 1, 1, 1])
	img = ax[0].imshow(np.zeros((IMG_SIZE_Y, IMG_SIZE_X)), cmap="gray", vmin=0, vmax=255)
	corners_plt, = ax[0].plot([0], [0], 'ro', markersize=1)
	board_corner_points = ax[0].scatter([0, 0, 0, 0], [0, 0, 0, 0], label="detected board corners", zorder=100)
	mask_plt = ax[0].imshow(mask, cmap=pr_cmap, vmin=0, vmax=1, alpha=0.25)

	static_corner_plt, = ax[0].plot([CORNER_BL[0], CORNER_BR[0], CORNER_TR[0], CORNER_TL[0]], [CORNER_BL[1], CORNER_BR[1], CORNER_TR[1], CORNER_TL[1]], 'go')

	slider = Slider(ax[1], 'blockSize', 0, 20, valinit=2, valstep=1)
	slider2 = Slider(ax[2], 'ksize', 1, 31, valinit=5, valstep=2)
	slider3 = Slider(ax[3], 'k', 0, 0.1, valinit=0.02)
	slider4 = Slider(ax[4], 'threshold', 0, 0.4, valinit=0.01)

	def update(_):
		with Timer("update"):
			frame, _ = consumer_conn.recv()
			corners = find_corners(frame, block_size=slider.val, ksize=slider2.val, k=slider3.val, threshold=slider4.val)
			corner_br, corner_bl, corner_tl, corner_tr = find_board_corners(frame)
			frame = process_frame(frame)

			img.set_array(frame)
			board_corner_points.set_offsets([corner_br, corner_bl, corner_tl, corner_tr])
			corners_plt.set_xdata(corners[:, 1])
			corners_plt.set_ydata(corners[:, 0])
			return img, board_corner_points, mask_plt, corners_plt, static_corner_plt

	anim = FuncAnimation(fig, update, cache_frame_data=False, interval=0, blit=True)

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


if __name__ == '__main__':
	main()
