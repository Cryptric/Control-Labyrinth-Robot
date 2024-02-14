from multiprocessing import Pipe, Process, Event

import cv2

import Davis346Reader


def main():
	consumer_conn, producer_conn = Pipe(False)
	termination_event = Event()
	p: Process = Process(target=Davis346Reader.run, args=(producer_conn, termination_event))
	p.start()
	while True:
		frame = consumer_conn.recv()
		cv2.imshow("frame", frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			termination_event.set()
			break


if __name__ == "__main__":
	main()
