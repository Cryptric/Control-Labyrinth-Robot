from multiprocessing import Pipe, Process, Event
from multiprocessing.connection import Connection
from functools import partial
from typing import List

import matplotlib

from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib.animation import FuncAnimation
from matplotlib.image import AxesImage
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch import Tensor

import CueNetV2
from CueNetV2 import device
import Davis346Reader
from utils.Plotting import pr_cmap
from utils.FrameUtils import remove_distortion

matplotlib.use('TkAgg')


IMG_SIZE_X = 346
IMG_SIZE_Y = 260

PROCESSING_SIZE_WIDTH = 240
PROCESSING_SIZE_HEIGHT = 180

X_EDGE = IMG_SIZE_X - PROCESSING_SIZE_WIDTH
Y_EDGE = IMG_SIZE_Y - PROCESSING_SIZE_HEIGHT

PROCESSING_X = X_EDGE // 2
PROCESSING_Y = Y_EDGE // 2


def calc_position_heatmap(frame1: Tensor, frame2: Tensor, frame3: Tensor, cue_net: CueNetV2) -> np.ndarray:
	frame_stack = torch.unsqueeze(torch.stack((frame1, frame2, frame3)), 0)
	with torch.no_grad():
		output = cue_net(frame_stack)
	return output.cpu().detach().numpy()


def update(_, consumer_conn: Connection, frame_buffer: List[np.ndarray], cue_net: CueNetV2, img: AxesImage, pos_heatmap: AxesImage):
	if consumer_conn.poll():
		try:
			frame = consumer_conn.recv()
			frame = remove_distortion(frame)
			frame += 64
			frame[frame < 64] = 255
			img.set_array(frame)
			new_frame = torch.squeeze(TF.to_tensor(frame[PROCESSING_Y:PROCESSING_Y + PROCESSING_SIZE_HEIGHT, PROCESSING_X:PROCESSING_X + PROCESSING_SIZE_WIDTH].astype("float32") / 255).to(device))
			if len(frame_buffer) >= 2:
				heatmap = calc_position_heatmap(frame_buffer.pop(0), frame_buffer[0], new_frame, cue_net)
				heatmap = np.pad(heatmap[0], ((Y_EDGE // 2, Y_EDGE // 2), (X_EDGE // 2, X_EDGE // 2)))
				pos_heatmap.set_array(heatmap)
				vmin, vmax = heatmap.min(), heatmap.max()
				pos_heatmap.set_clim(vmin=vmin, vmax=vmax)
			frame_buffer.append(new_frame)
		except EOFError:
			print("Producer exited")
			print("Shutting down")
			return None
	return img, pos_heatmap


def main():
	cue_net = CueNetV2.load_cue_net_v2()
	frame_buffer = []

	fig, ax = plt.subplots()
	img = ax.imshow(np.zeros((IMG_SIZE_Y, IMG_SIZE_X)), cmap="gray", vmin=0, vmax=255)
	pos_heatmap = ax.imshow(np.zeros((IMG_SIZE_Y, IMG_SIZE_X)), cmap=pr_cmap, alpha=1)

	processing_section_marker = patches.Rectangle((PROCESSING_X, PROCESSING_Y), PROCESSING_SIZE_WIDTH, PROCESSING_SIZE_HEIGHT, linewidth=1, edgecolor='r', facecolor='none')
	ax.add_patch(processing_section_marker)

	consumer_conn, producer_conn = Pipe(False)
	termination_event = Event()
	p = Process(target=Davis346Reader.run, args=(producer_conn, termination_event))
	p.start()
	producer_conn.close()

	update_func = partial(update, consumer_conn=consumer_conn, frame_buffer=frame_buffer, cue_net=cue_net, img=img, pos_heatmap=pos_heatmap)
	_anim = FuncAnimation(fig, update_func, cache_frame_data=False, interval=0)

	plt.show()
	print("plot terminated, sending termination event")
	termination_event.set()
	p.join()


if __name__ == "__main__":
	main()
