"""
Convert a Silent-Face-Anti-Spoofing ONNX model to OpenVINO IR.

Source repo: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
License: Apache 2.0 (commercial-safe).

How to get an ONNX file:
    1. Clone the repo above.
    2. Use their provided conversion notebook / `convert_to_onnx.py` to export
       one of the pretrained checkpoints (e.g. `2.7_80x80_MiniFASNetV2.pth`)
       to ONNX. Use the matching `--as_scale` (2.7 or 4.0) at inference time.

Usage:
    python tools/export_antispoof.py path/to/model.onnx path/to/output_dir
"""
import sys
from pathlib import Path

import openvino as ov


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    onnx_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    model = ov.convert_model(str(onnx_path))
    out_xml = out_dir / (onnx_path.stem + '.xml')
    ov.save_model(model, str(out_xml))
    print(f'Saved IR to {out_xml}')


if __name__ == '__main__':
    main()
