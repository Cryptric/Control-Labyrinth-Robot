from HighLevelController.NearestPointController import NearestPointController
from Labyrinth.LabyrinthDetection import detect_labyrinth
from Params import *
from utils.ControlUtils import *


class PathFindingNearestPointController(NearestPointController):
	def __init__(self, frame):
		frame_cut = frame[CORNER_TL[1]:CORNER_BR[1], CORNER_TL[0]:CORNER_BR[0]]
		ball_pos = find_center4(frame_cut)
		frame_cut, bframe, patched_frame, bmframe, bmcframe, with_walls, lbfs, detected_walls, circ_locs_x, circ_locs_y, hole_positions, G, path, path_weights, path_idx_x, path_idx_y = detect_labyrinth(frame_cut, ball_pos)

		path_idx_x += CORNER_TL[0]
		path_idx_y += CORNER_TL[1]

		coordinate_transform_mat, mm2px_mat = get_transform_matrices()
		path_mm = sequence_apply_inverse_transform(mm2px_mat, path_idx_x, path_idx_y).T

		super().__init__(path_mm)
