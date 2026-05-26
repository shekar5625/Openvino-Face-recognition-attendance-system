"""Quick discriminative test: does the IR model produce different outputs
for different inputs? If yes, conversion is fine. If no, conversion is broken."""
import os
import sys
import numpy as np
import cv2
import openvino as ov

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
IR_PATH = os.path.join(PROJECT, 'models', 'models', 'anti_spoof',
                       '2.7_80x80_MiniFASNetV2.xml')
GALLERY = os.path.join(PROJECT, 'my_gallery', 'my_gallery')


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def run(model, img):
    # img is HWC BGR uint8, any size
    img = cv2.resize(img, (80, 80)).astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)[None]
    return softmax(model(img)[0].flatten())


core = ov.Core()
model = core.compile_model(IR_PATH, 'CPU')

# 1. Random noise
np.random.seed(0)
noise = (np.random.rand(80, 80, 3) * 255).astype(np.uint8)
print('random noise:        ', run(model, noise))

# 2. Solid black
print('black:               ', run(model, np.zeros((80, 80, 3), np.uint8)))

# 3. Solid white
print('white:               ', run(model, np.full((80, 80, 3), 255, np.uint8)))

# 4. First gallery image (real face)
for f in sorted(os.listdir(GALLERY)):
    if f.split('.')[-1].lower() in ('jpg', 'jpeg', 'png'):
        img = cv2.imread(os.path.join(GALLERY, f))
        print('gallery {:20s}'.format(f), run(model, img))
        break

# 5. Heavily blurred gallery image (mimics phone screen blur)
blurred = cv2.GaussianBlur(img, (21, 21), 0)
print('gallery blurred:     ', run(model, blurred))

# 6. Inverted colors (definitely shouldn't look real)
inverted = 255 - img
print('inverted:            ', run(model, inverted))
