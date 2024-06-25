import re
import subprocess


DAVIS_CONNECTED_REGEX = r"\d*-\d*\s*[\da-f]*:[\da-f]*\s*DAVIS\sBLUE\s346"
DAVIS_ATTACHED_REGEX = r"\d*-\d*\s*[\da-f]*:[\da-f]*\s*DAVIS\sBLUE\s346\s*Attached"

ARDUINO_CONNECTED_REGEX = r"\d*-\d*\s*[\da-f]*:[\da-f]*\s*USB\-SERIAL CH340 \(COM4\)"
ARDUINO_ATTACHED_REGEX = r"\d*-\d*\s*[\da-f]*:[\da-f]*\s*USB\-SERIAL CH340 \(COM4\)\s*Attached"


def check_device_attached(usbipd_list_output, regex) -> bool:
	return bool(re.search(regex, usbipd_list_output))


def check_device_connected(usbipd_list_output, regex) -> bool:
	return bool(re.search(regex, usbipd_list_output))


def get_device_busid(usbipd_list_output, regex):
	davis_status_line = re.search(regex, usbipd_list_output).group()
	return re.search(r"\d*-\d*", davis_status_line).group()

def connect_device(device_name, usbipd_list_output, connected_regex, attached_regex):
	device_connected = check_device_connected(usbipd_list_output, connected_regex)
	if device_connected:
		davis_attached = check_device_attached(usbipd_list_output, attached_regex)
		if not davis_attached:
			device_busid = get_device_busid(usbipd_list_output, connected_regex)
			print("Found {} at busid {}".format(device_name, device_busid))
			subprocess.run(["usbipd", "attach", "--wsl", "--busid", device_busid])
			print("device attached")
		else:
			print("{} already attached".format(device_name))
	else:
		print("{} not connected!".format(device_name))
		# exit(-1)

def main():
	output = subprocess.run(["usbipd", "list"], stdout=subprocess.PIPE)
	usbipd_list_output = output.stdout.decode('utf-8')
	connect_device("DAVIS", usbipd_list_output, DAVIS_CONNECTED_REGEX, DAVIS_ATTACHED_REGEX)
	connect_device("ARDUINO", usbipd_list_output, ARDUINO_CONNECTED_REGEX, ARDUINO_ATTACHED_REGEX)


if __name__ == '__main__':
	main()
