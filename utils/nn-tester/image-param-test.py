from functools import partial
from multiprocessing import Pipe, Event, Process

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
import torchvision.transforms.functional as TF
import CueNetV2
import Davis346Reader
from CueNetV2 import device
from Params import Y_EDGE, X_EDGE, IMG_SIZE_Y, IMG_SIZE_X, PROCESSING_Y, PROCESSING_SIZE_HEIGHT, PROCESSING_X, \
	PROCESSING_SIZE_WIDTH
from utils.FrameUtils import remove_distortion



def main():
	cue_net = CueNetV2.load_cue_net_v2()
	cue_net.warmup()

	frame_buffer = []
	consumer_conn, producer_conn = Pipe(False)
	termination_event = Event()
	p = Process(target=Davis346Reader.run, args=(producer_conn, termination_event))
	p.start()
	producer_conn.close()

	fig, ax = plt.subplots(nrows=4)
	slider = Slider(ax[2], 'Bias', 0, 255, valinit=0, valstep=1)
	slider2 = Slider(ax[3], 'Gain', 0.0, 5.0, valinit=1.0)

	img = ax[0].imshow(np.zeros((IMG_SIZE_Y, IMG_SIZE_X)), cmap="gray", vmin=0, vmax=255)
	pos_heatmap = ax[1].imshow(np.zeros((IMG_SIZE_Y, IMG_SIZE_X)), cmap='inferno', alpha=1)

	def update(_, img_plt, pos_heatmap_plt):
		def process_frame(pframe):
			pframe = remove_distortion(pframe)
			# frame += 96
			# frame[frame < 96] = 255
			pframe = np.clip(pframe.astype(np.uint16) * slider2.val + slider.val, 0, 255)
			return pframe, torch.squeeze(TF.to_tensor(pframe[PROCESSING_Y:PROCESSING_Y + PROCESSING_SIZE_HEIGHT, PROCESSING_X:PROCESSING_X + PROCESSING_SIZE_WIDTH].astype("float32") / 255).to(device))

		frame, _ = consumer_conn.recv()
		frame, new_frame = process_frame(frame)
		if len(frame_buffer) == 3:
			heatmap = cue_net.calc_position_heatmap(new_frame, frame_buffer.pop(0), frame_buffer[0])
			heatmap = np.pad(heatmap[0], ((Y_EDGE // 2, Y_EDGE // 2), (X_EDGE // 2, X_EDGE // 2)))
			img_plt.set_array(frame)
			pos_heatmap_plt.set_array(heatmap)
			vmin, vmax = heatmap.min(), heatmap.max()
			pos_heatmap_plt.set_clim(vmin=vmin, vmax=vmax)

		frame_buffer.append(new_frame)
		return img_plt, pos_heatmap_plt

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
