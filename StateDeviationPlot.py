import pickle
import time

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
from scipy.ndimage import gaussian_filter
plt.rcParams.update({'font.size': 24})


# angle/unit vector function from stackoverflow https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
def unit_vector(vector):
	""" Returns the unit vector of the vector.  """
	return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
	""" Returns the angle in radians between vectors 'v1' and 'v2'  """
	v1_u = unit_vector(v1)
	v2_u = unit_vector(v2)
	return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def main():
	with open("recorded_x.pkl", "rb") as f:
		recoded_data_x = pickle.load(f)

	with open("recorded_y.pkl", "rb") as f:
		recoded_data_y = pickle.load(f)

	ref_pred = np.array([0, 1])

	n = len(recoded_data_x) - 1

	ref_points = []
	actual_points = []
	outliers = 0

	for i in range(n):
		state_x, ref_x, pred_x = recoded_data_x[i]
		state_y, ref_y, pred_y = recoded_data_y[i]

		state_x_ppi = recoded_data_x[i + 1][0]
		state_y_ppi = recoded_data_y[i + 1][0]

		pos = np.array([state_x[0], state_y[0]])
		pred_pos = np.array([pred_x[0], pred_y[0]])
		actual_pos = np.array([state_x_ppi[0], state_y_ppi[0]])

		pred_pos = pred_pos - pos
		actual_pos = actual_pos - pos

		pred_pos_l2 = np.linalg.norm(pred_pos)

		pred_pos = pred_pos / pred_pos_l2
		actual_pos = actual_pos / pred_pos_l2

		angle = angle_between(ref_pred, pred_pos)
		angle = - angle
		R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

		pred_pos = R @ pred_pos
		actual_pos = R @ actual_pos

		if np.linalg.norm(pred_pos - ref_pred) > 0.1:
			pred_pos = R.T @ pred_pos
			actual_pos = R.T @ actual_pos

			angle = - angle
			R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

			pred_pos = R @ pred_pos
			actual_pos = R @ actual_pos

		if np.linalg.norm(pred_pos - actual_pos) <= 2:
			ref_points.append(pred_pos)
			actual_points.append(actual_pos)
		else:
			outliers += 1

	ref_points = np.array(ref_points)
	actual_points = np.array(actual_points)
	print(f"removed {outliers} outliers fram {n} datapoints")

	def scatter():
		plt.scatter(actual_points[:, 0], actual_points[:, 1], color="green", label="actual position")

	def heatmap():
		heatmap, xedges, yedges = np.histogram2d(actual_points[:, 0], actual_points[:, 1], bins=500, range=[[-1, 1], [-0.5, 1.5]])
		heatmap = gaussian_filter(heatmap, sigma=3)
		extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

		plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='inferno')

	heatmap()

	plt.scatter(ref_pred[0], ref_pred[1], color="blue", label="normalized predicted position", s=200)
	plt.scatter([0], [0], color="red", label="current position", s=200)

	plt.grid()
	plt.legend()
	plt.ylim(-0.5, 1.5)
	plt.xlim(-1, 1)
	plt.show()


if __name__ == '__main__':
	main()
