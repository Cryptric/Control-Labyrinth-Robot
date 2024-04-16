import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

CueNetV3_path = "Nets/pt-labi_CNN.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CueNetV2(nn.Module):
	def __init__(self):
		super(CueNetV2, self).__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.ReLU(),
			nn.BatchNorm2d(64))

		self.layer2 = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.ReLU(),
			nn.BatchNorm2d(64))

		self.layer3 = nn.Sequential(
			nn.MaxPool2d(2, stride=2, padding=0))

		self.layer4 = nn.Sequential(
			nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.ReLU(),
			nn.BatchNorm2d(128))

		self.layer5 = nn.Sequential(
			nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.ReLU(),
			nn.BatchNorm2d(128))

		self.layer6 = nn.Sequential(
			nn.MaxPool2d(2, stride=2, padding=0))

		self.layer7 = nn.Sequential(
			nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.ReLU(),
			nn.BatchNorm2d(256))

		self.layer8 = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.ReLU(),
			nn.BatchNorm2d(256))

		self.layer9 = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.ReLU(),
			nn.BatchNorm2d(256))

		self.layer10 = nn.UpsamplingBilinear2d(scale_factor=2)

		self.layer11 = nn.Sequential(
			nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.ReLU(),
			nn.BatchNorm2d(128))

		self.layer12 = nn.Sequential(
			nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.ReLU(),
			nn.BatchNorm2d(128))

		self.layer13 = nn.Sequential(
			nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.ReLU(),
			nn.BatchNorm2d(128))

		self.layer14 = nn.UpsamplingBilinear2d(scale_factor=2)

		self.layer15 = nn.Sequential(
			nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.ReLU(),
			nn.BatchNorm2d(64))

		self.layer16 = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.ReLU(),
			nn.BatchNorm2d(64))

		self.layer17 = nn.Sequential(
			nn.Conv2d(64, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.ReLU(),
			nn.BatchNorm2d(1))

		self.layer18 = nn.Softmax(dim=1)

	def forward(self, x):
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.layer5(x)
		x = self.layer6(x)
		x = self.layer7(x)
		x = self.layer8(x)
		x = self.layer9(x)
		x = self.layer10(x)
		x = self.layer11(x)
		x = self.layer12(x)
		x = self.layer13(x)
		x = self.layer14(x)
		x = self.layer15(x)
		x = self.layer16(x)
		x = self.layer17(x)
		x = x.view(1, -1)
		x = self.layer18(x)
		x = x.reshape(1, 180, 240)

		return x

	def calc_position_heatmap(self, frame1: Tensor, frame2: Tensor, frame3: Tensor) -> np.ndarray:
		frame_stack = torch.unsqueeze(torch.stack((frame1, frame2, frame3)), 0)
		with torch.no_grad():
			output = self(frame_stack)
		return output.cpu().detach().numpy()

	def warmup(self):
		frame_stack = torch.zeros((1, 3, 180, 240)).to(device)
		with torch.no_grad():
			_o = self(frame_stack)


def load_cue_net_v2() -> CueNetV2:
	net_state = torch.load(CueNetV3_path)
	net = CueNetV2()
	net.load_state_dict(net_state["state_dict"])
	net.to(device)
	net.eval()
	return net
