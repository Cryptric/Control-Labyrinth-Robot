from functools import partial
from multiprocessing import Queue

import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D

from Params import *
from utils.ControlUtils import *
from utils.FrameUtils import *
from utils.Plotting import pr_cmap

# plt.rcParams.update({'font.size': 8})


def onclick(event):
	print("clicked")




def plot(data_queue, termination_event, line_plot_labels, corner_br, corner_bl, corner_tl, corner_tr):
	plt.rcParams.update({'font.size': 12})
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


	fig, ax = plt.subplots(nrows=3, height_ratios=[3, 1, 1])
	fig.set_size_inches(12, 8)

	ax[0].set_xticks([])
	ax[0].set_yticks([])
	ax[0].set_xlabel("X position")
	ax[0].set_ylabel("Y position")

	img = ax[0].imshow(np.zeros((IMG_SIZE_Y, IMG_SIZE_X)), cmap="gray", vmin=0, vmax=255)
	pos_heatmap = ax[0].imshow(np.zeros((IMG_SIZE_Y, IMG_SIZE_X)), cmap=pr_cmap, alpha=1, zorder=99)

	ball_pos_plot, = ax[0].plot([], [], marker='o', label="Past ball position", markersize=5, c="gray")

	ball_pos_patch = ax[0].add_patch(plt.Circle((10, 10), 2, color="red", label="Current ball position", zorder=100))

	fig.canvas.mpl_connect('button_press_event', partial(onclick))

	ref_trajectory_plot, = ax[0].plot([], [], marker="x", label="Reference trajectory", markersize=5, c="orange")
	pred_trajectory_plot, = ax[0].plot([], [], marker="x", label="Predicted MPC trajectory", markersize=5, c="green")

	ax[1].set_xticks([])
	ax[1].set_yticks([-30, 0, 30])
	ax[1].set_title("Control signal")
	ax[1].set_xlabel("Time")
	ax[1].set_ylabel("[°]")
	ax[1].grid()
	line_plots = []
	for label in line_plot_labels:
		line, = ax[1].plot([1, 2], [1, 2], label=label)
		line_plots.append(line)
	ax[1].set_ylim((U_min - DISTURBANCE_INTEGRAL_CLIP/min(K_x, K_y)) * 180 / np.pi * 2, (U_max + DISTURBANCE_INTEGRAL_CLIP/min(K_x, K_y)) * 180 / np.pi * 2)

	ax[2].set_xticks([])
	ax[2].set_title("Ball velocity")
	speed_plot_x,  = ax[2].plot([], [], label="Ball velocity x")
	speed_plot_y,  = ax[2].plot([], [], label="Ball velocity y")
	ax[2].set_ylim(-300, 300)
	ax[2].set_xlabel("Time")
	ax[2].set_ylabel("[mm/s]")

	def update(_, speed_plot, ax2, ax3):
		frame, heatmap, pos, target_trajectory, [pred_state_x, pred_state_y], data_points, speed, time = data_queue.get()
		while not data_queue.empty():
			frame, heatmap, pos, target_trajectory, [pred_state_x, pred_state_y], data_points, speed, time = data_queue.get()

		img.set_array(remove_distortion(frame))

		# pos_heatmap.set_array(heatmap)
		# vmin, vmax = heatmap.min(), heatmap.max()
		# pos_heatmap.set_clim(vmin=vmin, vmax=vmax)

		pos = apply_transform(mm2px_mat, pos)
		target_trajectory = sequence_apply_transform(mm2px_mat, target_trajectory[:, 0], target_trajectory[:, 1])
		pred_trajectory = sequence_apply_transform(mm2px_mat, pred_state_x, pred_state_y)

		ball_pos_plot.set_xdata(np.append(ball_pos_plot.get_xdata(), pos[0])[-50:])
		ball_pos_plot.set_ydata(np.append(ball_pos_plot.get_ydata(), pos[1])[-50:])

		ball_pos_patch.set_center([pos[0], pos[1]])

		ref_trajectory_plot.set_xdata(target_trajectory[0])
		ref_trajectory_plot.set_ydata(target_trajectory[1])

		pred_trajectory_plot.set_xdata(pred_trajectory[0])
		pred_trajectory_plot.set_ydata(pred_trajectory[1])

		for i in range(len(line_plots)):
			line_plots[i].set_xdata(np.append(line_plots[i].get_xdata(), time)[-50:])
			line_plots[i].set_ydata(np.append(line_plots[i].get_ydata(), data_points[i])[-50:])
		ax2.relim()
		ax2.autoscale_view()

		for speed_line, data_point in zip(speed_plot, speed):
			speed_line.set_xdata(np.append(speed_line.get_xdata(), time)[-50:])
			speed_line.set_ydata(np.append(speed_line.get_ydata(), data_point)[-50:])
		ax3.relim()
		ax3.autoscale_view()

		return img, pos_heatmap, ball_pos_plot, ref_trajectory_plot, pred_trajectory_plot, ball_pos_patch, *line_plots, *speed_plot

	fig.legend()
	fig.tight_layout()
	update_func = partial(update, speed_plot=[speed_plot_x, speed_plot_y], ax2=ax[1], ax3=ax[2])
	anim = FuncAnimation(fig, update_func, cache_frame_data=False, interval=0, blit=True)

	plt.grid()
	plt.show()

	print("plot terminated, sending termination event")
	termination_event.set()
