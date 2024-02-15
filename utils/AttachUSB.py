import re
import subprocess


DAVIS_CONNECTED_REGEX = r"\d*-\d*\s*[\da-f]*:[\da-f]*\s*DAVIS\sBLUE\s346"
DAVIS_ATTACHED_REGEX = r"\d*-\d*\s*[\da-f]*:[\da-f]*\s*DAVIS\sBLUE\s346\s*Attached"


def check_davis_attached(usbipd_list_output) -> bool:
	return bool(re.search(DAVIS_ATTACHED_REGEX, usbipd_list_output))


def check_davis_connected(usbipd_list_output) -> bool:
	return bool(re.search(DAVIS_CONNECTED_REGEX, usbipd_list_output))


def get_davis_busid(usbipd_list_output) -> str:
	davis_status_line = re.search(DAVIS_CONNECTED_REGEX, usbipd_list_output).group()
	return re.search(r"\d*-\d*", davis_status_line).group()


def main():
	output = subprocess.run(["usbipd", "list"], stdout=subprocess.PIPE)
	usbipd_list_output = output.stdout.decode('utf-8')
	davis_connected = check_davis_connected(usbipd_list_output)
	if davis_connected:
		davis_attached = check_davis_attached(usbipd_list_output)
		if not davis_attached:
			davis_busid = get_davis_busid(usbipd_list_output)
			print("Found DAVIS at busid {}".format(davis_busid))
			subprocess.run(["usbipd", "attach", "--wsl", "--busid", davis_busid])
			print("DAVIS attached")
		else:
			print("DAVIS already attached")
	else:
		print("DAVIS camera not connected!")
		exit(-1)


if __name__ == '__main__':
	main()
