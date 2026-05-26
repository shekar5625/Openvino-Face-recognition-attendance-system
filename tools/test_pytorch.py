"""Verify PyTorch model gives different outputs for different inputs."""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import cv2

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
REPO = PROJECT / 'third_party' / 'Silent-Face-Anti-Spoofing'
CKPT = REPO / 'resources' / 'anti_spoof_models' / '2.7_80x80_MiniFASNetV2.pth'
sys.path.insert(0, str(REPO))
from src.model_lib.MiniFASNet import MiniFASNetV2

model = MiniFASNetV2(conv6_kernel=(5, 5))
state = torch.load(str(CKPT), map_location='cpu', weights_only=True)
state = {k.replace('module.', ''): v for k, v in state.items()}
model.load_state_dict(state)
model.eval()


def run(img):
    img = cv2.resize(img, (80, 80)).astype(np.float32)  # NO /255
    t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
    with torch.no_grad():
        out = model(t)
        probs = F.softmax(out, dim=1).numpy().flatten()
    return probs


np.random.seed(0)
print('random noise:', run((np.random.rand(80, 80, 3) * 255).astype(np.uint8)))
print('black:       ', run(np.zeros((80, 80, 3), np.uint8)))
print('white:       ', run(np.full((80, 80, 3), 255, np.uint8)))

gallery = PROJECT / 'my_gallery' / 'my_gallery'
for f in sorted(gallery.iterdir()):
    if f.suffix.lower() in ('.jpg', '.jpeg', '.png'):
        img = cv2.imread(str(f))
        print(f'gallery {f.name}:', run(img))
        break
