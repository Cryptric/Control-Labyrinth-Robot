from datetime import datetime
from multiprocessing import Pipe, Process, Event, Queue

import numpy as np
import serial

import matplotlib

import Davis346Reader
import Plotter
from HighLevelController.NearestPointController import NearestPointController
from HighLevelController.PathFindingNearestPointController import PathFindingNearestPointController
from analysis.FollowingScoreCalculator import calculate_mean_variance
from controllers.Controllers import *
from utils.ControlUtils import *
from utils.FrameUtils import *
from utils.store import store
matplotlib.use("tkAgg")


def main():
	start_time = datetime.now()

	# Connect arduino
	arduino = serial.Serial('/dev/ttyUSB0', 115200, timeout=5)
	time.sleep(1)

	# Initialize camera process
	consumer_conn, producer_conn = Pipe(False)
	termination_event = Event()
	p = Process(target=Davis346Reader.run, args=(producer_conn, termination_event))
	p.start()
	producer_conn.close()

	# Initialize plot process
	plot_queue = Queue()
	plot_process = Process(target=Plotter.plot, args=(plot_queue, termination_event, ["Servo angle control signal x", "Servo angle control signal y"], CORNER_BR, CORNER_BL, CORNER_TL, CORNER_TR))
	plot_process.start()

	# Preprocessing for controller
	preprocessing_frame = None
	for i in range(10):
		preprocessing_frame, _ = consumer_conn.recv()

	fig, ax = plt.subplots()
	fig.tight_layout()
	ax.set_title("Select the center of the ball")
	ax.axis('off')
	ax.imshow(preprocessing_frame, cmap='gray', vmin=0, vmax=255)

	new_pattern = np.zeros((10, 10), dtype=np.uint8)
	new_pattern_subpixel = np.zeros((40, 40), dtype=np.uint8)

	def on_click(event):
		x = round(event.xdata)
		y = round(event.ydata)
		ball_pattern = preprocessing_frame[y-5:y+5, x-5:x+5]
		new_pattern[:] = ball_pattern
		new_pattern_subpixel[:] = cv2.resize(pattern, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
		plt.close()
	fig.canvas.mpl_connect('button_press_event', on_click)
	plt.show()
	print("ball selection done")

	coordinate_transform_mat, mm2px_mat = get_transform_matrices()

	orig_focal_x_pos = np.array([0, 0], dtype=np.float64)
	orig_focal_y_pos = np.array([0, 0], dtype=np.float64)
	orig_focal_x_pos[:] = detect_focal_x(preprocessing_frame)
	orig_focal_y_pos[:] = detect_focal_y(preprocessing_frame)
	orig_focal_pos = np.array([orig_focal_x_pos[0], orig_focal_y_pos[1]])
	prev_angles = np.array([0, 0], dtype=np.float64)

	ball_pos = find_center4(preprocessing_frame, p=new_pattern, sp=new_pattern_subpixel)
	prev_pos = np.array(apply_transform(coordinate_transform_mat, calc_corrected_pos(ball_pos, 0, 0)))

	# path_controller = NearestPointController(gen_path_medium_labyrinth())
	# path_controller = NearestPointController(gen_path_simple_labyrinth_auto())
	path_controller = NearestPointController(gen_path_custom_labyrinth_auto())

	controller = LinearMPC(prev_pos, path_controller)
	#controller = SimulationController(prev_pos)

	# clear pipe
	while consumer_conn.poll():
		consumer_conn.recv()

	runtime_start = time.time()

	recorded_data_x = {"state": [], "delay_compensated_state": [], "target_trajectory": [], "predicted_state": [], "mpc_signal": [], "signal_multiplier": [], "disturbance_compensation": [], "delta": [], "board_angle": []}
	recorded_data_y = {"state": [], "delay_compensated_state": [], "target_trajectory": [], "predicted_state": [], "mpc_signal": [], "signal_multiplier": [], "disturbance_compensation": [], "delta": [], "board_angle": []}
	frames = []

	controller.set_recorders(recorded_data_x, recorded_data_y)

	def update():
		if consumer_conn.poll():
			try:
				with Timer("control loop"):
					frame, t = consumer_conn.recv()
					x_mm, y_mm, angle_x, angle_y = detect_ball_pos_mm(frame, orig_focal_pos, coordinate_transform_mat, prev_angles, legacy=False, mm2px_mat=mm2px_mat, p=new_pattern, sp=new_pattern_subpixel)
					if x_mm is None:
						return

					speed_x, speed_y = calc_speed(x_mm, prev_pos[0], dt), calc_speed(y_mm, prev_pos[1], dt)
					prev_pos[:] = [x_mm, y_mm]

					(sig_x_rad, sig_y_rad), (sig_mult_x, sig_mult_y), (dist_x, dist_y), (pred_xk_x, pred_xk_y), target_trajectory = controller([x_mm, y_mm], [speed_x, speed_y])

					signal_x_deg = np.rad2deg((sig_x_rad[STEPS_DEAD_TIME] - dist_x / K_x) * sig_mult_x)
					signal_y_deg = np.rad2deg((sig_y_rad[STEPS_DEAD_TIME] - dist_y / K_y) * sig_mult_y)

					send_control_signal(arduino, signal_x_deg, signal_y_deg)

				plot_queue.put_nowait((frame, None, [x_mm, y_mm], target_trajectory, [pred_xk_x, pred_xk_y], [signal_x_deg, signal_y_deg], [speed_x, speed_y], t))
				controller.visualization_update()
				# recording
				recorded_data_x["state"].append([x_mm, speed_x])
				recorded_data_x["delay_compensated_state"].append(pred_xk_x[STEPS_DEAD_TIME])
				recorded_data_x["target_trajectory"].append(target_trajectory[0:N, 0])
				recorded_data_x["predicted_state"].append(pred_xk_x)
				recorded_data_x["mpc_signal"].append(sig_x_rad)
				recorded_data_x["signal_multiplier"].append(sig_mult_x)
				recorded_data_x["disturbance_compensation"].append(dist_x)
				recorded_data_x["board_angle"].append(angle_x)

				recorded_data_y["state"].append([y_mm, speed_y])
				recorded_data_y["delay_compensated_state"].append(pred_xk_y[STEPS_DEAD_TIME])
				recorded_data_y["target_trajectory"].append(target_trajectory[0:N, 1])
				recorded_data_y["predicted_state"].append(pred_xk_y)
				recorded_data_y["mpc_signal"].append(sig_y_rad)
				recorded_data_y["signal_multiplier"].append(sig_mult_y)
				recorded_data_y["disturbance_compensation"].append(dist_y)
				recorded_data_y["board_angle"].append(angle_y)

				if RECORD_FRAMES:
					frames.append(frame)

			except EOFError:
				print("Producer exited")
				print("Shutting down")
				termination_event.set()

	# control loop
	while not termination_event.is_set():
		update()

	controller.destroy()

	# make sure pipe isn't full, such that producer can exit
	while p.is_alive():
		try:
			if consumer_conn.poll():
				consumer_conn.recv()
		except EOFError:
			break
	consumer_conn.close()

	p.join()
	plot_process.join()

	while not plot_queue.empty():
		plot_queue.get()

	plot_queue.close()

	with open("recorded_x.pkl", 'wb') as f:
		pickle.dump(recorded_data_x, f, pickle.HIGHEST_PROTOCOL)

	with open("recorded_y.pkl", 'wb') as f:
		pickle.dump(recorded_data_y, f, pickle.HIGHEST_PROTOCOL)

	print_timers()
	follow_mse = calc_following_mse(recorded_data_x) + calc_following_mse(recorded_data_y)
	store(recorded_data_x, recorded_data_y, start_time, time.time() - runtime_start, follow_mse, frames)
	mean, variance = calculate_mean_variance(recorded_data_x, recorded_data_y)
	print(f"Mean: {mean}, Variance: {variance}")
	exit(0)


if __name__ == "__main__":
	main()
