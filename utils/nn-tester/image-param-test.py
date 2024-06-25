from functools import partial
from multiprocessing import Pipe, Event, Process

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import patches
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
import torchvision.transforms.functional as TF
import CueNetV2
import Davis346Reader
from CueNetV2 import device
from Params import *
from utils.ControlUtils import find_center, find_center2
from utils.FrameUtils import remove_distortion
from utils.Plotting import pr_cmap

matplotlib.use("TkAgg")


def main():
	cue_net = CueNetV2.load_cue_net_v2()
	cue_net.warmup()

	frame_buffer = []
	consumer_conn, producer_conn = Pipe(False)
	termination_event = Event()
	p = Process(target=Davis346Reader.run, args=(producer_conn, termination_event))
	p.start()
	producer_conn.close()

	fig, ax = plt.subplots(nrows=4, height_ratios=[5, 5, 1, 1])
	slider = Slider(ax[2], 'Bias', 0, 255, valinit=0, valstep=1)
	slider2 = Slider(ax[3], 'Gain', 0.0, 8.0, valinit=1.0)

	img = ax[0].imshow(np.zeros((IMG_SIZE_Y, IMG_SIZE_X)), cmap="gray", vmin=0, vmax=255)
	pos_heatmap_overlay = ax[0].imshow(np.zeros((IMG_SIZE_Y, IMG_SIZE_X - 2)), cmap=pr_cmap, alpha=1)
	pos_heatmap = ax[1].imshow(np.zeros((IMG_SIZE_Y, IMG_SIZE_X - 2)), cmap='inferno', alpha=1)

	pos_patch = plt.Circle((0, 0), 7, fill=False, color='purple', zorder=101)
	pos_plt = ax[0].add_patch(pos_patch)

	# processing_section_marker = patches.Rectangle((PROCESSING_X, PROCESSING_Y), PROCESSING_SIZE_WIDTH, PROCESSING_SIZE_HEIGHT, linewidth=1, edgecolor='r', facecolor='none')
	# processing_region = ax[0].add_patch(processing_section_marker)

	def update(_, img_plt, pos_heatmap_plt):
		def process_frame(pframe):
			pframe = remove_distortion(pframe)
			#pframe = cv2.equalizeHist(pframe)
			pframe = np.clip(pframe.astype(np.uint16) * slider2.val + slider.val, 0, 255).astype(np.uint8)
			return pframe, torch.squeeze(TF.to_tensor(pframe.astype("float32") / 255).to(device))

		frame, _ = consumer_conn.recv()
		frame, new_frame = process_frame(frame)
		if len(frame_buffer) == 3:
			heatmap = cue_net.calc_position_heatmap(new_frame, frame_buffer.pop(0), frame_buffer[0])[0]
			x, y = find_center2(heatmap)
			print(f"Found ball at x={x}, y={y}")
			# heatmap = np.pad(heatmap[0], ((Y_EDGE // 2, Y_EDGE // 2), (X_EDGE // 2, X_EDGE // 2)))
			img_plt.set_array(frame)
			pos_heatmap_overlay.set_array(heatmap)
			pos_heatmap_plt.set_array(heatmap)
			vmin, vmax = heatmap.min(), heatmap.max()
			pos_heatmap_overlay.set_clim(vmin, vmax)
			pos_heatmap_plt.set_clim(vmin=vmin, vmax=vmax)

		frame_buffer.append(new_frame)

		argmax_pos = np.unravel_index(frame.argmax(), frame.shape)
		pos_plt.set_center(argmax_pos[::-1])

		return img_plt, pos_heatmap_plt, pos_heatmap_overlay, pos_plt

	update_func = partial(update, img_plt=img, pos_heatmap_plt=pos_heatmap)
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


if __name__ == '__main__':
	main()
