import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, gaussian_filter
from skimage.draw import line

from Labyrinth.LabyrinthDetection import gen_circ_pattern
from utils.ControlUtils import gen_path_custom_labyrinth2, gen_circ

walls = np.array([[[0, 138], [95, 143]],
					[[90, 96], [95, 143]],
					[[43, 205], [48, 191]],
					[[43, 205], [159, 200]],
					[[159, 227], [164, 111]],
					[[164, 180], [180, 185]],
					[[137, 116], [164, 111]],
					[[137, 116], [142, 57]],
					[[89, 57], [142, 62]],
					[[206, 187], [233, 192]],
					[[228, 192], [233, 139]],
					[[205, 134], [210, 86]],
					[[184, 91], [210, 86]],
					[[184, 91], [189, 22]],
					[[159, 27], [214, 22]],
					[[159, 27], [164, 0]]])

holes = np.array([[32, 185],
	[10, 48],
	[150, 190],
	[150, 56],
	[196, 120],
	[171, 170],
	[219, 177],
	[220, 133],
	[219, 59],
	[265, 89],
	[242, 14],
	[80, 12]])

max_x = 273
max_y = 227


def generate_base_map():
	print(max_x, max_y)

	labyrinth_map = np.ones((max_y, max_x), dtype=np.uint8)
	for wall in walls:
		xs = sorted(wall[:, 0])
		ys = sorted(wall[:, 1])
		labyrinth_map[ys[0]:ys[1], xs[0]:xs[1]] = 0

	circ_pattern = gen_circ_pattern(7)
	pattern_size_2 = int(circ_pattern.shape[0] / 2)
	for hole in holes:
		labyrinth_map[hole[1]-pattern_size_2:hole[1]+pattern_size_2+1, hole[0]-pattern_size_2:hole[0]+pattern_size_2+1] = circ_pattern

	return labyrinth_map


def path_into_image(path, labyrinth_map, score_dec=1):
	path_image = labyrinth_map.astype(np.uint16)
	progress_val = 4000
	for i in range(path.shape[0] - 1):
		[start_x, start_y] = path[i].astype(int)
		[end_x, end_y] = path[i + 1].astype(int)
		rr, cc = line(start_y, start_x, end_y, end_x)
		path_image[rr, cc] = progress_val
		progress_val -= score_dec
	return path_image


def outgrow_weights(labyrinth_map):
	while 1 in labyrinth_map:
		no_wall_map = labyrinth_map.copy()
		neighborhood_max = np.max(np.stack([
				np.pad(no_wall_map, [(1, 0), (0, 0)])[:-1, :],
				np.pad(no_wall_map, [(0, 1), (0, 0)])[1:, :],
				np.pad(no_wall_map, [(0, 0), (1, 0)])[:, :-1],
				np.pad(no_wall_map, [(0, 0), (0, 1)])[:, 1:]]),
			axis=0)
		reachable_mask = (labyrinth_map == 1)
		new_weight_mask = reachable_mask & (neighborhood_max != 1)
		labyrinth_map[new_weight_mask] = neighborhood_max[new_weight_mask] + 1



def labyrinth_score():
	labyrinth_map = generate_base_map()
	path = gen_path_custom_labyrinth2()
	labyrinth_map = path_into_image(path, labyrinth_map, score_dec=2)

	outgrow_weights(labyrinth_map)

	return labyrinth_map


def gen_neg_gradient_field():
	score_map = labyrinth_score()
	smooth_pre = score_map.astype(np.float32)
	smooth_pre[smooth_pre == 0] = np.max(smooth_pre)
	smooth_cost = gaussian_filter(smooth_pre, sigma=3)
	smooth_cost[score_map == 0] = np.nan

	dy, dx = np.gradient(smooth_cost, edge_order=2)

	dxy = np.stack((dx, dy), axis=2)
	norms = np.linalg.norm(dxy, axis=2)
	dy = dy / norms
	dx = dx / norms

	dx *= -1
	dy *= -1
	return dx, dy


def main():
	score_map = labyrinth_score()

	# np.savetxt("score3.csv", score_map, delimiter=",")

	Y, X = np.mgrid[0:score_map.shape[0], 0:score_map.shape[1]]

	score_map_nan = score_map.astype(np.float32)
	score_map_nan[score_map_nan == 0] = np.nan

	fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
	ax.plot_surface(X, Y, score_map_nan, cmap="magma")
	ax.set_xlabel("Position X [mm]")
	ax.set_ylabel("Position Y [mm]")
	ax.set_zlabel("Cost value")
	ax.set_title("Cost map")

	fig2, ax2 = plt.subplots()
	cost_plt = ax2.imshow(score_map_nan, cmap="magma", origin="lower")
	ax2.set_xlabel("Position X [mm]")
	ax2.set_ylabel("Position Y [mm]")
	ax2.set_title("Cost map")
	fig2.colorbar(cost_plt, label="Cost value", ax=ax2)

	fig3, ax3 = plt.subplots()
	ax3.set_title("Smoothed")
	smooth_pre = score_map.astype(np.float32)
	smooth_pre[smooth_pre == 0] = np.max(smooth_pre)
	smooth_cost = gaussian_filter(smooth_pre, sigma=3)
	smooth_cost[score_map == 0] = np.nan
	smooth_cost_plt = ax3.imshow(smooth_cost, cmap="magma", origin="lower")
	fig3.colorbar(smooth_cost_plt, label="Cost value", ax=ax3)


	dy, dx = np.gradient(smooth_cost, edge_order=2)
#
	dxy = np.stack((dx, dy), axis=2)
	norms = np.linalg.norm(dxy, axis=2)
	dy = dy / norms
	dx = dx / norms
#
	dx *= -1
	dy *= -1
#
	fig_g, ax_g = plt.subplots()
	ax_g.quiver(X, Y, dx, dy, scale=1, scale_units='xy')

	plt.show()


if __name__ == '__main__':
	main()
