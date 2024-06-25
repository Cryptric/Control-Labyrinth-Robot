import os

import cv2
import numpy as np
from PIL import Image
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.widgets import Button, Slider
from natsort import natsorted

from utils.FrameUtils import remove_distortion

matplotlib.use("tkAgg")


def load_image(path):
	img = Image.open(path)
	img.load()
	data = np.asarray(img, dtype=np.uint8)
	return data


def preprocess_img(img):
	pframe = remove_distortion(img)
	pframe = cv2.equalizeHist(pframe)
	return pframe


def load_images(path):
	files = [(path + f) for f in os.listdir(path) if f.endswith(".jpg")]
	files = natsorted(files)
	images = [load_image(f) for f in files]
	return files, images


def get_labels(images):
	fig = plt.figure()
	gs = fig.add_gridspec(2, 3, width_ratios=[1, 3, 1], height_ratios=[20, 1])
	gs.update(left=0.1, right=0.9, top=0.99, bottom=0.01, wspace=0.3, hspace=0)
	ax_img = fig.add_subplot(gs[0, :])
	ax_prev = fig.add_subplot(gs[1, 0])
	ax_index = fig.add_subplot(gs[1, 1])
	ax_next = fig.add_subplot(gs[1, 2])

	ax_img.set_xticks([])
	ax_img.set_yticks([])

	ball_radius = 4

	labels = np.zeros((len(images), 2))

	img_plt = ax_img.imshow(images[0], cmap='grey', vmin=0, vmax=255)
	label_patch = plt.Circle(labels[0], ball_radius, fill=False, color='r')
	label_plt = ax_img.add_patch(label_patch)
	btn_prev = Button(ax_prev, "Previous")
	btn_next = Button(ax_next, "Next")
	slider_index = Slider(ax_index, "idx", valmin=0, valmax=len(images)-1, valstep=1, valinit=0)

	def show_img(new_index, from_slider=False):
		if new_index >= len(images):
			print("Done labeling")
			return
		if new_index < 0:
			print("Illegal index")
			return
		img_plt.set_array(images[new_index])
		label_plt.set_center(labels[new_index])
		if not from_slider:
			slider_index.set_val(new_index)
		fig.canvas.draw()
		fig.canvas.flush_events()

	def set_label(event):
		ax_idx = -1
		for i, ax in enumerate([ax_img, ax_prev, ax_index, ax_next]):
			if ax == event.inaxes:
				ax_idx = i
		if ax_idx != 0:
			return
		x = event.xdata
		y = event.ydata
		labels[slider_index.val] = np.array([x, y])
		show_img(slider_index.val + 1)

	btn_prev.on_clicked(lambda e:  show_img(slider_index.val - 1))
	btn_next.on_clicked(lambda e:  show_img(slider_index.val + 1))
	fig.canvas.mpl_connect('button_press_event', set_label)
	slider_index.on_changed(lambda v: show_img(v, from_slider=True))

	plt.show()
	return labels


def store_labels(file_names, labels, base_path):
	file_name = f"{base_path}labels.csv"
	with open(file_name, 'a') as f:
		for i in range(len(file_names)):
			f.write(f"{file_names[i]},{labels[i, 0]},{labels[i, 1]}\n")



def main():
	path = "MLData/"
	file_names, images = load_images(path)
	processed_images = [preprocess_img(img) for img in images]
	labels = get_labels(processed_images)
	store_labels(file_names, labels, path)


if __name__ == '__main__':
	main()
