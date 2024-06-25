import matplotlib.pyplot as plt
import numpy as np
import cv2
from pyaer import libcaer
from pyaer.davis import DAVIS

from Labyrinth.LabyrinthDetection import *
from Params import *
from utils.FrameUtils import *
from utils.ControlUtils import *


def capture_labyrinth_image():
	device = DAVIS(noise_filter=True)
	device.start_data_stream()
	device.set_bias_from_json("davis346_config.json")

	print("Press 'c' to select a frame for calibration or 'q' to quit")

	frame = np.zeros((IMG_SIZE_Y, IMG_SIZE_X), dtype=np.uint8)
	while True:
		try:
			data = device.get_event()
			if data is not None:
				(_, _, _, _, _, frames, _, _) = data
				if (not isinstance(frames, int)) and frames.shape[0] != 0:
					frame = remove_distortion(frames[0])
					cv2.imshow('frame', frame)
					action = cv2.waitKey(1) & 0xFF
					if action == ord('c'):
						device.shutdown()
						return frame
					if action == ord('q'):
						print("Quitting")
						device.shutdown()
						exit(0)
		except KeyboardInterrupt:
			device.shutdown()
			exit(-1)


def show(frame, path):
	fig, ax = plt.subplots()
	ax.imshow(frame, cmap='gray')
	ax.plot(path[0], path[1], label="Path")

	plt.show()


def main():
	frame = capture_labyrinth_image()
	frame_cut = frame[CORNER_TL[1]:CORNER_BR[1], CORNER_TL[0]:CORNER_BR[0]]
	ball_pos = find_center4(frame_cut)
	frame_cut, bframe, patched_frame, bmframe, bmcframe, with_walls, lbfs, detected_walls, circ_locs_x, circ_locs_y, hole_positions, G, path, path_weights, path_idx_x, path_idx_y = detect_labyrinth(frame_cut, ball_pos)
	plot_all(frame_cut, bframe, patched_frame, bmframe, bmcframe, with_walls, lbfs, detected_walls, circ_locs_x, circ_locs_y, hole_positions, G, path, path_weights, path_idx_x, path_idx_y)
	path_idx_x += CORNER_TL[0]
	path_idx_y += CORNER_TL[1]

	coordinate_transform_mat, mm2px_mat = get_transform_matrices()
	path_mm = sequence_apply_inverse_transform(mm2px_mat, path_idx_x, path_idx_y).T
	path_px = sequence_apply_transform(mm2px_mat, path_mm[:, 0], path_mm[:, 1]).T

	show(frame, [path_px[:, 0], path_px[:, 1]])


if __name__ == '__main__':
	main()
