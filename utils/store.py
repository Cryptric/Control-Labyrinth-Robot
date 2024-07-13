import pickle

from StateDeviationPlot import statistics, axis_deviation
from inputimeout import inputimeout


def store(data_x, data_y, start_date, runtime, follow_mse, frames):
	try:
		comment = inputimeout(prompt="Comment: ", timeout=10)
	except:
		comment = None
	n = len(data_x)
	deviation_x = axis_deviation(data_x)
	deviation_y = axis_deviation(data_y)
	mu_x, std_x = statistics(deviation_x)
	mu_y, std_y = statistics(deviation_y)
	file_name = f"data_{str(start_date).replace(':', '-').replace(' ', '_').split('.0')[0]}.pkl"

	with open("./store/index.csv", "a") as index:
		index.write(f"{file_name},{start_date},{runtime},{comment},{n},{follow_mse},{mu_x},{std_x},{mu_y},{std_y}\n")

	with open(f"./store/{file_name}", "wb") as file:
		pickle.dump((data_x, data_y), file)

	if len(frames) != 0:
		with open(f"./store/{file_name.replace('data_', 'frames_')}", "wb") as file:
			pickle.dump(frames, file)

	print("Updated data store")
