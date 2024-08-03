import pickle

import numpy as np
import pandas as pd


def get_distances(positions, path):
	n = len(positions)
	distances = []
	for i in range(n):
		position = positions[i]
		path_distance = np.linalg.norm(path - position, axis=1)
		min_distance = np.min(path_distance)
		distances.append(min_distance)
	return np.array(distances)


def calculate_following_mean(distances):
	return np.mean(distances)


def calculate_following_variance(distances):
	return np.var(distances)


def calculate_mean_variance(data_x, data_y):
	states_x = np.array(data_x["state"])
	states_y = np.array(data_y["state"])
	positions = np.stack((states_x[:, 0], states_y[:, 0]), axis=1)

	target_trajectories_x = np.array(data_x["target_trajectory"])
	target_trajectories_y = np.array(data_y["target_trajectory"])

	target_trajectory = np.stack((target_trajectories_x[:, 0], target_trajectories_y[:, 0]), axis=1)

	distances = get_distances(positions, target_trajectory)
	mean = calculate_following_mean(distances)
	variance = calculate_following_variance(distances)

	return mean, variance

def main():
	file_name_0 = "2024-07-30_09-21-49.926217.pkl"
	index_table = pd.read_csv("store/index.csv")
	file_name = index_table["file_name"].iloc[-1]
	if file_name.startswith("data_"):
		file_name = file_name[5:]
	file = file_name
	with open(f"store/data_{file}", "rb") as f:
		data_x, data_y = pickle.load(f)
	mean, variance = calculate_mean_variance(data_x, data_y)
	print(f"Mean: {mean}, Variance: {variance}")


if __name__ == "__main__":
	main()
