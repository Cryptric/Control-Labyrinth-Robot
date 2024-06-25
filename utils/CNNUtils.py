import torch
import torchvision.transforms.functional as TF

from CueNetV2 import device
from Params import *


def frame2torch(frame):
	return torch.squeeze(TF.to_tensor(frame.astype("float32") / 255).to(device))
