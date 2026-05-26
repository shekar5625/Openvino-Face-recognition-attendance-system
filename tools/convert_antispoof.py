"""
End-to-end: Silent-Face-Anti-Spoofing .pth -> ONNX -> OpenVINO IR.

Prereqs:
  pip install torch torchvision
  git clone https://github.com/minivision-ai/Silent-Face-Anti-Spoofing \
      third_party/Silent-Face-Anti-Spoofing

Run (from `Openvino recognition` dir):
  python tools/convert_antispoof.py

Outputs: models/models/anti_spoof/2.7_80x80_MiniFASNetV2.xml (+ .bin)
"""
import sys
from pathlib import Path

import torch
import openvino as ov

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
REPO = PROJECT / 'third_party' / 'Silent-Face-Anti-Spoofing'
CKPT = REPO / 'resources' / 'anti_spoof_models' / '2.7_80x80_MiniFASNetV2.pth'
OUT_DIR = PROJECT / 'models' / 'models' / 'anti_spoof'

if not REPO.exists():
    sys.exit(f'Repo not found at {REPO}. Clone it first (see file header).')
if not CKPT.exists():
    sys.exit(f'Checkpoint not found at {CKPT}.')

sys.path.insert(0, str(REPO))
from src.model_lib.MiniFASNet import MiniFASNetV2  # noqa: E402

OUT_DIR.mkdir(parents=True, exist_ok=True)

model = MiniFASNetV2(embedding_size=128, conv6_kernel=(5, 5),
                     drop_p=0.0, num_classes=3, img_channel=3)
state = torch.load(str(CKPT), map_location='cpu', weights_only=True)
print('state type:', type(state))
print('checkpoint key sample:', list(state.keys())[:3])
print('model    key sample:', list(model.state_dict().keys())[:3])
# Their checkpoints are saved with a "module." prefix from DataParallel.
state = {k.replace('module.', ''): v for k, v in state.items()}
result = model.load_state_dict(state, strict=False)
print('missing keys   :', result.missing_keys[:5], '... total:', len(result.missing_keys))
print('unexpected keys:', result.unexpected_keys[:5], '... total:', len(result.unexpected_keys))
model.eval()

onnx_path = OUT_DIR / '2.7_80x80_MiniFASNetV2.onnx'
dummy = torch.randn(1, 3, 80, 80)
torch.onnx.export(model, dummy, str(onnx_path),
                  input_names=['input'], output_names=['logits'],
                  opset_version=13, dynamo=False)
print(f'ONNX:  {onnx_path}')

ir = ov.convert_model(str(onnx_path))
xml_path = OUT_DIR / '2.7_80x80_MiniFASNetV2.xml'
ov.save_model(ir, str(xml_path))
print(f'IR:    {xml_path}')
