"""
 Silent-Face-Anti-Spoofing wrapper (minivision-ai, Apache 2.0).

 The model expects an 80x80 BGR crop taken from a bbox expanded by
 `scale` (typically 2.7 or 4.0 depending on the checkpoint) around the
 face center. Output is 3 logits — index 1 = "real" after softmax.
"""

import logging as log

import cv2
import numpy as np

from ie_module import Module


class AntiSpoof(Module):
    def __init__(self, core, model, scale=2.7, threshold=0.5):
        super().__init__(core, model, 'Anti-Spoofing')
        if len(self.model.inputs) != 1:
            raise RuntimeError("Anti-spoof model expects 1 input layer")
        self.input_tensor_name = self.model.inputs[0].get_any_name()
        self.input_shape = list(self.model.inputs[0].shape)
        self.nchw_layout = self.input_shape[1] == 3
        if self.nchw_layout:
            self.h, self.w = self.input_shape[2], self.input_shape[3]
        else:
            self.h, self.w = self.input_shape[1], self.input_shape[2]
        self.scale = scale
        self.threshold = threshold
        self.real_class = 1  # overridden by calibration
        self._valid = []

    def _crop(self, frame, roi):
        bw, bh = float(roi.size[0]), float(roi.size[1])
        cx = float(roi.position[0]) + bw / 2.0
        cy = float(roi.position[1]) + bh / 2.0
        side = max(bw, bh) * self.scale
        x1 = int(max(cx - side / 2.0, 0))
        y1 = int(max(cy - side / 2.0, 0))
        x2 = int(min(cx + side / 2.0, frame.shape[1]))
        y2 = int(min(cy + side / 2.0, frame.shape[0]))
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    def _preprocess_one(self, crop):
        # Silent-Face MiniFASNet was trained on raw [0,255] BGR — do NOT /255.
        # The BN running stats encode that scale (running_mean values in the tens).
        resized = cv2.resize(crop, (self.w, self.h)).astype(np.float32)
        if self.nchw_layout:
            resized = resized.transpose(2, 0, 1)
        return resized.reshape(self.input_shape)

    def start_async_batch(self, frame, rois):
        self.clear()
        self._valid = []
        for roi in rois:
            crop = self._crop(frame, roi)
            if crop is None or crop.size == 0:
                self._valid.append(False)
                continue
            self._valid.append(True)
            self.enqueue({self.input_tensor_name: self._preprocess_one(crop)})

    def get_probs(self):
        """Return softmax probabilities for each valid roi (None for invalid)."""
        outs = self.get_outputs()
        probs_list = []
        j = 0
        for valid in self._valid:
            if not valid:
                probs_list.append(None)
                continue
            logits = outs[j].flatten().astype(np.float64)
            e = np.exp(logits - logits.max())
            probs_list.append(e / e.sum())
            j += 1
        return probs_list

    def get_real_scores(self):
        scores = []
        for probs in self.get_probs():
            if probs is None:
                scores.append(0.0)
                continue
            log.debug('anti_spoof probs: %s (real_class=%d)',
                      np.array2string(probs, precision=3), self.real_class)
            idx = self.real_class if probs.size > self.real_class else 0
            scores.append(float(probs[idx]))
        return scores

    def infer_scores(self, frame, rois):
        self.start_async_batch(frame, rois)
        return self.get_real_scores()

    def infer_probs(self, frame, rois):
        self.start_async_batch(frame, rois)
        return self.get_probs()
