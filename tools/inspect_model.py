"""Inspect intermediate activations - find which layer kills the signal."""
import sys
from pathlib import Path
import numpy as np
import torch

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

# Quick stats on a BN buffer to spot weirdness
print('--- conv1.bn stats ---')
print('weight:        ', model.conv1.bn.weight.data[:5].tolist())
print('bias:          ', model.conv1.bn.bias.data[:5].tolist())
print('running_mean:  ', model.conv1.bn.running_mean.data[:5].tolist())
print('running_var:   ', model.conv1.bn.running_var.data[:5].tolist())

print('\n--- prob (final linear) weight stats ---')
w = model.prob.weight.data
print('shape:', w.shape, 'mean:', w.mean().item(), 'std:', w.std().item(),
      'min:', w.min().item(), 'max:', w.max().item())

# Run on two very different inputs, capture intermediate activations
acts = {}
def hook(name):
    def fn(_m, _i, o):
        acts[name] = o.detach()
    return fn

for n in ['conv1', 'conv_3', 'conv_4', 'conv_5', 'conv_6_dw', 'linear', 'bn', 'prob']:
    getattr(model, n).register_forward_hook(hook(n))

a = torch.zeros(1, 3, 80, 80)
b = torch.ones(1, 3, 80, 80)
with torch.no_grad():
    model(a)
    a_acts = {k: v.clone() for k, v in acts.items()}
    model(b)
    b_acts = {k: v.clone() for k, v in acts.items()}

print('\n--- diff between black vs white at each layer ---')
for n in ['conv1', 'conv_3', 'conv_4', 'conv_5', 'conv_6_dw', 'linear', 'bn', 'prob']:
    diff = (a_acts[n] - b_acts[n]).abs().mean().item()
    print(f'{n:12s} mean abs diff = {diff:.6e}, a-mean={a_acts[n].mean().item():.3e}, b-mean={b_acts[n].mean().item():.3e}')
