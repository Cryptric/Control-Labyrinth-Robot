import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation
from skimage.draw import line


def hole_cost(pos_x, pos_y, width, height, a=10):
	y, x = np.mgrid[0:height, 0:width]
	distances = np.sqrt((x - pos_x) ** 2 + (y - pos_y) ** 2)
	cost = 1.1 * ((0.12 * distances) ** a + 100) / ((0.12 * distances) ** a + 10) - 1.1
	return cost


def gen_hole_weight_frame(hole_positions, width, height):
	n = hole_positions.shape[0]
	weights = np.zeros((height, width))
	for i in range(n):
		weights += hole_cost(hole_positions[i, 0], hole_positions[i, 1], width, height)
	return weights


def add_boarder_wall(frame, width=3):
	return np.pad(frame, [(width, width), (width, width)], constant_values=0)


def hole_wall_connection(pos_x, pos_y, wall_mask, width, height, distance_threshold):
	wall_mask = wall_mask.copy()
	y, x = np.mgrid[0:height, 0:width]
	distances = np.sqrt((x - pos_x) ** 2 + (y - pos_y) ** 2)

	con_idx_y, con_idx_x = np.where((distances <= distance_threshold) & (wall_mask == 0))
	for i in range(con_idx_x.shape[0]):
		rr, cc = line(int(np.round(pos_y)), int(np.round(pos_x)), con_idx_y[i], con_idx_x[i])
		wall_mask[rr, cc] = 0
	return wall_mask


def hole_wal_connections(hole_positions, wall_mask, distance_threshold):
	width = wall_mask.shape[1]
	height = wall_mask.shape[0]
	n = hole_positions.shape[0]
	for i in range(n):
		wall_mask = hole_wall_connection(hole_positions[i, 0], hole_positions[i, 1], wall_mask, width, height, distance_threshold)
	return wall_mask


def wall_cost(pos_x, pos_y, width, height):
	y, x = np.mgrid[0:height, 0:width]
	distances = np.sqrt((x - pos_x) ** 2 + (y - pos_y) ** 2)
	cost = 10 / (distances + 1)
	return cost


def set_walls(weights, wall_mask, val, width, height, s=5):
	weights += gaussian_filter(20 * (1 - wall_mask), sigma=s)
	return weights


def clip_exp(weights):
	weights = np.power(10, weights)
	weights = np.clip(weights, 0, 500)
	return weights


def labyrinth_mask(weights):
	mask = np.zeros_like(weights)
	mask[weights >= 300] = -1
	return mask


def labyrinth_bfs(labyrinth, start_x=150, start_y=15):
	max_val = labyrinth.shape[0] * labyrinth.shape[1]
	labyrinth = labyrinth.copy()
	labyrinth[labyrinth != -1] = max_val
	labyrinth[start_y, start_x] = 1
	iteration = 1
	neighborhood_structure = np.array([[False, True, False], [True, True, True], [False, True, False]])
	while iteration in labyrinth:
		iter_mask = labyrinth == iteration
		iter_mask = binary_dilation(iter_mask, neighborhood_structure)
		iteration = iteration + 1
		step = labyrinth.copy()
		step[iter_mask] = iteration
		labyrinth = np.minimum(labyrinth, step)
	labyrinth[labyrinth == max_val] = -1
	return labyrinth


def path_weight(mask, s=3):
	weights = mask.copy()
	weights[weights == -1] = 255
	weights += gaussian_filter(weights, sigma=s)
	weights[mask == -1] = -1
	return weights


def plot_weight_frame(frame):
	fig, ax = plt.subplots()
	ax.imshow(frame, cmap='inferno')


def plot_bfs(bfs_frame):
	bfs_frame = bfs_frame.copy()
	bfs_frame[bfs_frame == -1] = -400
	fig, ax = plt.subplots()
	ax.imshow(bfs_frame, cmap='inferno')


def plot_path_weight(weights, path_idx_x, path_idx_y):
	fig, ax = plt.subplots()
	ax.imshow(weights, cmap='inferno')
	ax.plot(path_idx_x, path_idx_y)


def img_idx_2_node_idx(x, y, mask):
	mask = mask.copy()
	w, h = mask.shape[1], mask.shape[0]
	flat_idx = (y - 1) * w + x
	mask[mask == 0] = 1
	mask[mask == -1] = 0
	return int(np.sum(mask.flatten()[0:flat_idx]))



def gen_graph(mask):
	# https://stackoverflow.com/questions/67259669/make-graph-from-mask-grid-in-python
	ys, xs = np.where(mask == 0)
	distances = np.sqrt((ys - ys.reshape(-1, 1)) ** 2 + (xs - xs.reshape(-1, 1)) ** 2)
	adjacency_matrix = (distances <= 1).astype(np.int64)
	np.fill_diagonal(adjacency_matrix, 0)

	G = nx.from_numpy_array(adjacency_matrix)
	n = G.number_of_nodes()

	pos_attrs = {i: (xs[i], ys[i]) for i in range(n)}
	nx.set_node_attributes(G, pos_attrs, "pos")

	return G


def get_path(G, shortest_path):
	k = len(shortest_path)
	x = np.zeros(k)
	y = np.zeros(k)
	for i, u in enumerate(shortest_path):
		xs, ys = G.nodes[u]['pos']
		x[i] = xs
		y[i] = ys
	return x, y


def set_edge_weights(G, weight_matrix):
	weight_matrix = weight_matrix.flatten()
	weight_matrix = weight_matrix[weight_matrix != -1]
	for (u, v) in G.edges:
		G[u][v]['weight'] = (weight_matrix[u] + weight_matrix[v]) / 2


def plot_graph(G, path):
	fig, ax = plt.subplots()
	node_colors = ["blue" if n in path else "red" for n in G.nodes()]
	edges = G.edges()
	weights = [G[u][v]['weight'] for u, v in edges]
	nx.draw(G, nx.get_node_attributes(G, 'pos'), ax=ax, node_color=node_colors, width=2, edge_cmap=plt.cm.inferno, edge_color=weights)
