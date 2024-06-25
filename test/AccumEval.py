from os import listdir

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.widgets import Slider

base_path = "RecordTest"


def load_frame(file_index):
	img = Image.open(f"{base_path}/img-{file_index}.jpg")
	img.load()
	frame = np.asarray(img, dtype=np.uint8)
	return frame


def load_events(file_index):
	events = np.load(f"{base_path}/events-{file_index}.npy")
	dt = max(events[:, 0]) - min(events[:, 0])
	pos_events = events[events[:, 3] == 1]
	neg_events = events[events[:, 3] == 0]
	return pos_events, neg_events, dt


def distance_filter(events_pos, pos, max_distance):
	distances = np.sqrt((events_pos[:, 0] - pos[0]) ** 2 + (events_pos[:, 1] - pos[1]) ** 2)
	return events_pos[distances < max_distance]


def plot(file_index):
	frame = load_frame(file_index)

	fig, ax = plt.subplots(nrows=2, height_ratios=[10, 1])

	frame_plt = ax[0].imshow(frame, cmap='gray')

	pos_events, neg_events, _ = load_events(file_index)
	pe_plot, = ax[0].plot(pos_events[:, 1], pos_events[:, 2], 'bo')
	ne_plot, = ax[0].plot(neg_events[:, 1], neg_events[:, 2], 'ro')

	file_count = int(len(listdir(base_path))/2)
	frame_slider = Slider(ax[1], "Frame", 0, file_count-1, valstep=1, valinit=file_index)

	y = 20
	x = 183
	pos = np.array([x, y])

	label_patch = plt.Circle((x, y), 7, fill=False, color='orange', zorder=100)
	label_plt = ax[0].add_patch(label_patch)

	bri_patch = plt.Circle((x, y), 7, fill=False, color='purple', zorder=101)
	bri_plt = ax[0].add_patch(bri_patch)

	def update(_):
		cframe = load_frame(frame_slider.val)
		frame_plt.set_array(cframe)
		pe, ne, dt = load_events(frame_slider.val)
		# print(f"dt: {dt}us (? i guess)")
		pe_plot.set_xdata(pe[:, 1])
		pe_plot.set_ydata(pe[:, 2])
		ne_plot.set_xdata(ne[:, 1])
		ne_plot.set_ydata(ne[:, 2])

		bri_pos = np.unravel_index(cframe.argmax(), cframe.shape)
		bri_plt.set_center(bri_pos[::-1])

		events_pos = np.vstack([pe, ne])[:, 1:3]
		events_pos = distance_filter(events_pos, pos, 20)
		if len(events_pos) > 0:
			pos[:] = np.mean(events_pos, axis=0)
			label_plt.set_center(pos)

	frame_slider.on_changed(update)

	def onclick(event):
		if event.ydata >= 1:
			pos[0] = event.xdata
			pos[1] = event.ydata
			print(event.ydata)

	fig.canvas.mpl_connect('button_press_event', onclick)

	plt.show()


if __name__ == '__main__':
	plot(27)
