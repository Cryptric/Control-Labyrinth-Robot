import matplotlib
from matplotlib import pyplot as plt

from Labyrinth.LabyrinthSolver import *


def plot_all(frame, bframe, patched_frame, bmframe, bmcframe, with_walls, lbfs, detected_walls, circ_locs_x, circ_locs_y, hole_positions, G, path, path_weights, path_idx_x, path_idx_y):
    plot_weight_frame(with_walls)
    plot_bfs(lbfs)
    plot_wall_detection(detected_walls)
    plot_overview(frame, bframe, patched_frame, bmframe, bmcframe, with_walls, circ_locs_x, circ_locs_y, hole_positions)
    if G is not None:
        plot_graph(G, path)
        plot_path_weight(path_weights, path_idx_x, path_idx_y)


def plot_overview(frame, bframe, patched_frame, bmframe, bmcframe, with_walls, circ_locs_x, circ_locs_y, hole_positions):
    fig, ax = plt.subplots(nrows=4, ncols=2)

    ax[0, 0].imshow(frame, cmap='gray')
    ax[0, 1].hist(frame.flatten(), bins=256)

    ax[1, 0].imshow(bframe, cmap='gray')

    ax[1, 1].imshow(bframe, cmap='gray')
    ax[1, 1].scatter(circ_locs_x, circ_locs_y)
    ax[1, 1].scatter(hole_positions[:, 0] - 3, hole_positions[:, 1] - 3, color='r')

    ax[2, 0].imshow(patched_frame, cmap='gray')

    ax[2, 1].imshow(bmframe, cmap='gray')

    ax[3, 0].imshow(bmcframe, cmap='gray')

    ax[3, 1].imshow(with_walls, cmap='inferno')

    fig.tight_layout()


def plot_wall_detection(wall_frame):
    fig, ax = plt.subplots()
    ax.imshow(wall_frame)
