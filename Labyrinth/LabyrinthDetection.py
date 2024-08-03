import PIL.Image
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import cv2 as cv

from Labyrinth.LabyrinthSolver import *
from Labyrinth.Visualization import *
from Labyrinth.WallDetector import *


def load_img(path):
	img = PIL.Image.open(path)
	img.load()
	frame = np.asarray(img, dtype=np.uint8)
	frame_g = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	return frame_g


def cut(frame, x=30, y=7, w=280, h=238):
	return frame[y:y + h, x:x + w]


def apply_thresholding(frame, threshold):
	pframe = frame.copy()
	pframe[pframe < threshold] = 0
	pframe[pframe >= threshold] = 1
	return pframe


def apply_morphological_operations(frame, pattern_size=5):
	kernel = np.ones((pattern_size, pattern_size), np.uint8)
	frame = cv.morphologyEx(frame, cv.MORPH_CLOSE, kernel)
	frame = cv.morphologyEx(frame, cv.MORPH_OPEN, kernel)
	return frame


def gen_circ_pattern(r):
	r = int(r + 0.5)
	y, x = np.mgrid[-r:r + 1, -r:r + 1]
	bcirc = np.round(np.sqrt(x ** 2 + y ** 2)) >= r
	return bcirc.astype(np.uint8)


def match_circles(frame, r, threshold=0.63):
	frame = frame.copy()
	pattern = gen_circ_pattern(r)
	w, h = pattern.shape[::-1]
	res = cv.matchTemplate(frame, pattern, cv.TM_CCOEFF_NORMED)
	loc_y, loc_x = np.where(res >= threshold)
	loc_y, loc_x = loc_y + r, loc_x + r
	return loc_y, loc_x


def find_pattern_match_group(p_x, p_y, locs_x, locs_y, max_distance=10):
	distances = np.sqrt((locs_x - p_x) ** 2 + (locs_y - p_y) ** 2)
	selection_mask = distances <= max_distance
	return locs_x[selection_mask], locs_y[selection_mask]


def get_hole_positions(locs_x, locs_y):
	n = locs_x.shape[0]

	locations = set()

	for i in range(n):
		ps_x, ps_y = find_pattern_match_group(locs_x[i], locs_y[i], locs_x, locs_y)
		p_x = np.mean(ps_x)
		p_y = np.mean(ps_y)
		locations.add((p_x, p_y))
	return np.array([np.array(e) for e in locations])


def patch_holes(frame, locations, patch_r=8):
	frame = frame.copy()
	n = locations.shape[0]
	w, h = 2 * patch_r, 2 * patch_r
	for i in range(n):
		x, y = int(np.round(locations[i, 0] - patch_r)), max(int(np.round(locations[i, 1] - patch_r)), 0)
		frame[y: y + h, x: x + w] = 1
	return frame


def detect_labyrinth(frame, ball_pos, with_graph=True):
	start_x, start_y = 150, 15
	if ball_pos is not None:
		start_x, start_y = ball_pos[0], ball_pos[1]
	bframe = apply_thresholding(frame, 30)

	if ball_pos is not None:
		bframe = patch_holes(bframe, np.array([[start_x, start_y]]), patch_r=8)
	circ_locs_y, circ_locs_x = match_circles(bframe, 8)
	hole_positions = get_hole_positions(circ_locs_x, circ_locs_y)
	patched_frame = patch_holes(bframe, hole_positions)

	bmframe = apply_morphological_operations(patched_frame)
	bmframe = add_boarder_wall(bmframe, width=3)
	hole_positions = hole_positions + 3

	bmcframe = hole_wal_connections(hole_positions, bmframe, 18)

	# solving the labyrinth
	w = bmcframe.shape[1]
	h = bmcframe.shape[0]

	max_val = 10
	weights = gen_hole_weight_frame(hole_positions, w, h)

	with_walls = set_walls(weights, bmcframe, max_val, w, h)

	with_walls = clip_exp(with_walls)

	lmask = labyrinth_mask(with_walls)

	lbfs = labyrinth_bfs(lmask, start_x=start_x, start_y=start_y)

	detected_walls = detect_walls(bmframe)

	path_weights = path_weight(lmask)

	# path_weights_plt = path_weights.copy()
	# path_weights_plt[path_weights_plt == -1] = np.nan
	# plt.imshow(path_weights_plt, cmap='inferno')
	# plt.colorbar()
	# plt.gca().set_axis_off()
	# plt.savefig("PlotData/images/path-weights.png")
	# plt.show()

	end_pos = np.unravel_index(np.argmax(lbfs, axis=None), lbfs.shape)


	# plt_bfs_img = lbfs.copy()
	# plt_bfs_img[plt_bfs_img == -1] = np.nan
	# plt.imshow(plt_bfs_img, cmap="inferno")
	# plt.plot(start_x, start_y, 'bo', markersize=10)
	# plt.plot(end_pos[1], end_pos[0], 'r*', markersize=10)
	# plt.gca().set_axis_off()
	# plt.savefig("PlotData/images/bfs")
	# plt.show()

	start_node_idx = img_idx_2_node_idx(start_x, start_y, lmask)
	end_node_idx = img_idx_2_node_idx(end_pos[1], end_pos[0], lmask)

	G = None
	path = None
	path_idx_x = None
	path_idx_y = None
	if with_graph:
		G = gen_graph(lmask)
		set_edge_weights(G, path_weights)
		path = nx.shortest_path(G, source=start_node_idx, target=end_node_idx, weight='weight')
		path_idx_x, path_idx_y = get_path(G, path)

	return frame, bframe, patched_frame, bmframe, bmcframe, with_walls, lbfs, detected_walls, circ_locs_x, circ_locs_y, hole_positions, G, path, path_weights, path_idx_x, path_idx_y


def save_img(path, array):
	im = PIL.Image.fromarray(array)
	im.save(path)

def main():
	frame_o = load_img("TestImages/undistorted-full.png")
	frame = cut(frame_o)


	frame, bframe, patched_frame, bmframe, bmcframe, with_walls, lbfs, detected_walls, circ_locs_x, circ_locs_y, hole_positions, G, path, path_weights, path_idx_x, path_idx_y = detect_labyrinth(frame, None, with_graph=True)

	# plt.imshow(bframe, cmap='gray')
	# plt.scatter(hole_positions[:, 0] - 3, hole_positions[:, 1] - 3, color='r')
	# plt.gca().axis('off')
	# plt.savefig("PlotData/images/hole-positions.png")
	# plt.show()

	# save_img("PlotData/images/patched-holes.png", patched_frame * 255)

	plt.imshow(frame_o, cmap='gray')
	plt.plot(path_idx_x + 30, path_idx_y + 7, c='r')
	plt.gca().axis('off')
	plt.savefig("PlotData/images/path.png")
	#plt.show()

	#plot_all(frame, bframe, patched_frame, bmframe, bmcframe, with_walls, lbfs, detected_walls, circ_locs_x, circ_locs_y, hole_positions, G, path, path_weights, path_idx_x, path_idx_y)
#
	plt.show()


if __name__ == '__main__':
	main()
