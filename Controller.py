from multiprocessing import Pipe, Process

import cv2
import numpy as np

import Davis346Reader



def main():
	consumer_conn, producer_conn = Pipe(False)
	p: Process = Process(target=Davis346Reader.run, args=(producer_conn, ))
	p.start()
	
	while True:
		frame = consumer_conn.recv()
		cv2.imshow("frame", frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	

if __name__ == "__main__":
	main()