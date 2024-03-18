from functools import partial
from multiprocessing import Queue

import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D

from Params import *
from utils.FrameUtils import mapping_px2mm
from utils.Plotting import pr_cmap


def onclick(event, target_pos_queue):
	target_pos_queue.put((event.xdata, event.ydata))
	# board_coordinates = mapping_px2mm(px2mm_mat, [event.xdata, event.ydata])
	# print("Board coordinates: {}, {}".format(board_coordinates[0], board_coordinates[1]))
	print("clicked")


def update(_, data_queue: Queue, img: AxesImage, pos_heatmap: AxesImage, ball_pos_plot: Line2D, processing_region, corner_points_plt, ref_trajectory_plot, line_plots, ax2):
	frame, heatmap, pos, ref_trajectory, data_points, time = data_queue.get()
	while not data_queue.empty():
		frame, heatmap, pos, ref_trajectory, data_points, time = data_queue.get()

	img.set_array(frame)

	pos_heatmap.set_array(heatmap)
	vmin, vmax = heatmap.min(), heatmap.max()
	pos_heatmap.set_clim(vmin=vmin, vmax=vmax)

	ball_pos_plot.set_xdata(np.append(ball_pos_plot.get_xdata(), pos[0])[-50:])
	ball_pos_plot.set_ydata(np.append(ball_pos_plot.get_ydata(), pos[1])[-50:])

	ref_trajectory_plot.set_xdata(ref_trajectory[0])
	ref_trajectory_plot.set_ydata(ref_trajectory[1])

	for i in range(len(line_plots)):
		line_plots[i].set_xdata(np.append(line_plots[i].get_xdata(), time)[-50:])
		line_plots[i].set_ydata(np.append(line_plots[i].get_ydata(), data_points[i])[-50:])

	ax2.relim()
	ax2.autoscale_view()

	return img, pos_heatmap, ball_pos_plot, processing_region, corner_points_plt, ref_trajectory_plot, *line_plots


def plot(data_queue, termination_event, target_pos_queue, line_plot_labels, corner_br, corner_bl, corner_tl, corner_tr):
	fig, ax = plt.subplots(nrows=2)
	img = ax[0].imshow(np.zeros((IMG_SIZE_Y, IMG_SIZE_X)), cmap="gray", vmin=0, vmax=255)
	pos_heatmap = ax[0].imshow(np.zeros((IMG_SIZE_Y, IMG_SIZE_X)), cmap=pr_cmap, alpha=1)

	ball_pos_plot, = ax[0].plot([], [], marker='o', label="Ball position", markersize=2, c="gray")

	processing_section_marker = patches.Rectangle((PROCESSING_X, PROCESSING_Y), PROCESSING_SIZE_WIDTH, PROCESSING_SIZE_HEIGHT, linewidth=1, edgecolor='r', facecolor='none')
	processing_region = ax[0].add_patch(processing_section_marker)

	fig.canvas.mpl_connect('button_press_event', partial(onclick, target_pos_queue=target_pos_queue))
	corner_points_plt = ax[0].scatter([corner_br[0], corner_bl[0], corner_tl[0], corner_tr[0]], [corner_br[1], corner_bl[1], corner_tl[1], corner_tr[1]], label="detected board corners")

	ref_trajectory_plot, = ax[0].plot([], [], marker="x", label="Reference trajectory", markersize=2, c="orange")

	line_plots = []
	for label in line_plot_labels:
		line, = ax[1].plot([1, 2], [1, 2], label=label)
		line_plots.append(line)
	ax[1].set_ylim(U_min * 180 / math.pi * 1.2, U_max * 180 / math.pi * 1.2)

	update_func = partial(update, data_queue=data_queue, img=img, pos_heatmap=pos_heatmap, ball_pos_plot=ball_pos_plot, processing_region=processing_region, corner_points_plt=corner_points_plt, ref_trajectory_plot=ref_trajectory_plot, line_plots=line_plots, ax2=ax[1])
	anim = FuncAnimation(fig, update_func, cache_frame_data=False, interval=0, blit=True)

	plt.legend()
	plt.grid()
	plt.show()

	print("plot terminated, sending termination event")
	termination_event.set()
