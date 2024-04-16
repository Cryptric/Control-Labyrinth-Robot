from functools import partial
from multiprocessing import Pipe, Event, Process

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import Davis346Reader
from Params import Y_EDGE, X_EDGE, IMG_SIZE_Y, IMG_SIZE_X, PROCESSING_Y, PROCESSING_SIZE_HEIGHT, PROCESSING_X, \
	PROCESSING_SIZE_WIDTH, CORNER_MASK_MAX_Y, CORNER_MASK_MIN_X, CORNER_MASK_MAX_X, CORNER_MASK_MIN_Y
from utils.ControlUtils import Timer, print_timers
from utils.FrameUtils import process_frame, remove_distortion, find_board_corners
from utils.Plotting import pr_cmap


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
	corner_points = ax.scatter([0, 0, 0, 0], [0, 0, 0, 0], label="detected board corners")
	mask_plt = ax.imshow(mask, cmap=pr_cmap, vmin=0, vmax=1, alpha=0.25)

	def update(_, img_plt, corner_points_plt):
		with Timer("update"):
			frame, _ = consumer_conn.recv()
			corner_br, corner_bl, corner_tl, corner_tr = find_board_corners(frame)
			frame, _ = process_frame(frame)

			img_plt.set_array(frame)
			corner_points.set_offsets([corner_br, corner_bl, corner_tl, corner_tr])
			return img_plt, corner_points_plt, mask_plt

	update_func = partial(update, img_plt=img, corner_points_plt=corner_points)
	anim = FuncAnimation(fig, update_func, cache_frame_data=False, interval=0, blit=True)

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
