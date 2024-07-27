import serial
import tkinter as tk

from Params import *
from utils.ControlUtils import send_control_signal

root = tk.Tk()

canvas1 = tk.Canvas(root, width=400, height=300)
canvas1.pack()

entry1 = tk.Entry(root)
entry2 = tk.Entry(root)
canvas1.create_window(200, 100, window=entry1)
canvas1.create_window(200, 140, window=entry2)

arduino = serial.Serial('/dev/ttyUSB1', 115200, timeout=5)
print("arduino connected")  # connect to arduino


def servo():
	x = float(entry1.get())
	y = float(entry2.get())

	signal_x_deg = (x / K_x)
	signal_y_deg = (y / K_y)

	send_control_signal(arduino, signal_x_deg, signal_y_deg)


button1 = tk.Button(text='Servos move', command=servo)
canvas1.create_window(200, 180, window=button1)

root.mainloop()