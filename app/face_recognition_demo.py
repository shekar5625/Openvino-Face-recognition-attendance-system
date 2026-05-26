#!/usr/bin/env python3
"""
 Copyright (c) 2018-2024 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import collections
import logging as log
import sys
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from openvino import Core, get_version

sys.path.append(str(Path(__file__).resolve().parents[1] / 'common_python'))
sys.path.append(str(Path(__file__).resolve().parents[1] / 'common_python/model_zoo'))

from utils import crop
from landmarks_detector import LandmarksDetector
from face_detector import FaceDetector
from faces_database import FacesDatabase
from face_identifier import FaceIdentifier
from anti_spoof import AntiSpoof
from attendance import AttendanceLogger

import monitors
from helpers import resolution
from images_capture import open_images_capture

from model_api.models import OutputTransform
from model_api.performance_metrics import PerformanceMetrics

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)

DEVICE_KINDS = ['CPU', 'GPU', 'HETERO']
SPOOF_ID = -2


def build_argparser():
    parser = ArgumentParser()

    general = parser.add_argument_group('General')
    general.add_argument('-i', '--input', required=True,
                         help='Required. An input to process. The input must be a single image, '
                              'a folder of images, video file or camera id.')
    general.add_argument('--loop', default=False, action='store_true',
                         help='Optional. Enable reading the input in a loop.')
    general.add_argument('-o', '--output',
                         help='Optional. Name of the output file(s) to save. Frames of odd width or height can be truncated. See https://github.com/opencv/opencv/pull/24086')
    general.add_argument('-limit', '--output_limit', default=1000, type=int,
                         help='Optional. Number of frames to store in output. '
                              'If 0 is set, all frames are stored.')
    general.add_argument('--output_resolution', default=None, type=resolution,
                         help='Optional. Specify the maximum output window resolution '
                              'in (width x height) format. Example: 1280x720. '
                              'Input frame size used by default.')
    general.add_argument('--no_show', action='store_true',
                         help="Optional. Don't show output.")
    general.add_argument('--crop_size', default=(0, 0), type=int, nargs=2,
                         help='Optional. Crop the input stream to this resolution.')
    general.add_argument('--match_algo', default='HUNGARIAN', choices=('HUNGARIAN', 'MIN_DIST'),
                         help='Optional. Algorithm for face matching. Default: HUNGARIAN.')
    general.add_argument('-u', '--utilization_monitors', default='', type=str,
                         help='Optional. List of monitors to show initially.')

    gallery = parser.add_argument_group('Faces database')
    gallery.add_argument('-fg', default='', help='Optional. Path to the face images directory.')
    gallery.add_argument('--run_detector', action='store_true',
                         help='Optional. Use Face Detection model to find faces '
                              'on the face images, otherwise use full images.')
    gallery.add_argument('--allow_grow', action='store_true',
                         help='Optional. Allow to grow faces gallery and to dump on disk. '
                              'Available only if --no_show option is off.')

    models = parser.add_argument_group('Models')
    models.add_argument('-m_fd', type=Path, required=True,
                        help='Required. Path to an .xml file with Face Detection model.')
    models.add_argument('-m_lm', type=Path, required=True,
                        help='Required. Path to an .xml file with Facial Landmarks Detection model.')
    models.add_argument('-m_reid', type=Path, required=True,
                        help='Required. Path to an .xml file with Face Reidentification model.')
    models.add_argument('-m_as', type=Path, default=None,
                        help='Optional. Path to an .xml file with the Silent-Face '
                             'Anti-Spoofing model. When supplied, faces classified '
                             'as spoof are excluded from voting and labeled SPOOF.')
    models.add_argument('--fd_input_size', default=(0, 0), type=int, nargs=2,
                        help='Optional. Specify the input size of detection model for '
                             'reshaping. Example: 500 700.')

    infer = parser.add_argument_group('Inference options')
    infer.add_argument('-d_fd', default='CPU', choices=DEVICE_KINDS,
                       help='Optional. Target device for Face Detection model. '
                            'Default value is CPU.')
    infer.add_argument('-d_lm', default='CPU', choices=DEVICE_KINDS,
                       help='Optional. Target device for Facial Landmarks Detection '
                            'model. Default value is CPU.')
    infer.add_argument('-d_reid', default='CPU', choices=DEVICE_KINDS,
                       help='Optional. Target device for Face Reidentification '
                            'model. Default value is CPU.')
    infer.add_argument('-v', '--verbose', action='store_true',
                       help='Optional. Be more verbose.')
    infer.add_argument('-t_fd', metavar='[0..1]', type=float, default=0.6,
                       help='Optional. Probability threshold for face detections.')
    infer.add_argument('-t_id', metavar='[0..1]', type=float, default=0.3,
                       help='Optional. Cosine distance threshold between two vectors '
                            'for face identification.')
    infer.add_argument('-exp_r_fd', metavar='NUMBER', type=float, default=1.15,
                       help='Optional. Scaling ratio for bboxes passed to face recognition.')
    infer.add_argument('--no_enroll_augment', action='store_true',
                       help='Optional. Disable gallery enrollment augmentation '
                            '(flip + brightness). Augmentation is on by default '
                            'and helps 1-shot accuracy.')
    infer.add_argument('--smooth_window', type=int, default=10,
                       help='Optional. Temporal smoothing: number of recent '
                            'frames to vote over per face track. Set 0 to disable.')
    infer.add_argument('--smooth_min_votes', type=int, default=5,
                       help='Optional. Temporal smoothing: minimum matching '
                            'votes within the window required to confirm an identity.')
    infer.add_argument('--min_face_size', type=int, default=60,
                       help='Optional. Faces smaller than this (pixels, shorter '
                            'side) are excluded from voting. Set 0 to disable.')
    infer.add_argument('--min_blur_var', type=float, default=40.0,
                       help='Optional. Laplacian variance threshold for blur. '
                            'Frames below this are excluded from voting. 0 disables.')
    infer.add_argument('-d_as', default='CPU', choices=DEVICE_KINDS,
                       help='Optional. Target device for the Anti-Spoofing model.')
    infer.add_argument('--as_scale', type=float, default=2.7,
                       help='Optional. Bbox expansion ratio for the anti-spoof crop. '
                            'Use the scale that matches the checkpoint (2.7 or 4.0).')
    infer.add_argument('--as_threshold', type=float, default=0.5,
                       help='Optional. Minimum real-class probability to accept a '
                            'face as live. Higher = stricter anti-spoof.')

    att = parser.add_argument_group('Attendance')
    att.add_argument('--db_url', default='sqlite:///attendance.db',
                     help='Optional. Attendance DB URL. Examples: '
                          '"sqlite:///attendance.db" (default), '
                          '"postgres://user:pass@host/db" (future). '
                          'Set to empty string to disable attendance logging.')
    att.add_argument('--snapshot_dir', default='logs/snapshots',
                     help='Optional. Directory to store attendance face snapshots.')
    return parser


class TemporalSmoother:
    """Per-track majority vote over a sliding window of recent identifications.

    Tracks are associated frame-to-frame by bbox IoU. A face is reported as a
    given identity only once that identity wins enough votes in the window —
    cuts single-frame false positives that plague 1-shot attendance systems.
    """
    def __init__(self, window=10, min_votes=5, iou_thresh=0.3):
        self.window = window
        self.min_votes = min_votes
        self.iou_thresh = iou_thresh
        self.tracks = []
        self.frame_num = 0

    @staticmethod
    def _iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    def update(self, rois, identities, quality_ok=None, spoof_flags=None):
        self.frame_num += 1
        if quality_ok is None:
            quality_ok = [True] * len(rois)
        if spoof_flags is None:
            spoof_flags = [False] * len(rois)
        smoothed = []
        smoothed_spoof = []
        used = set()
        for roi, ident, ok, raw_spoof in zip(rois, identities, quality_ok, spoof_flags):
            bbox = (float(roi.position[0]), float(roi.position[1]),
                    float(roi.position[0] + roi.size[0]),
                    float(roi.position[1] + roi.size[1]))
            best_iou, best_idx = 0.0, -1
            for i, tr in enumerate(self.tracks):
                if i in used:
                    continue
                iou = self._iou(bbox, tr['bbox'])
                if iou > best_iou:
                    best_iou, best_idx = iou, i
            if best_iou > self.iou_thresh:
                tr = self.tracks[best_idx]
                used.add(best_idx)
            else:
                tr = {
                    'votes': collections.deque(maxlen=self.window),
                    'spoof_votes': collections.deque(maxlen=self.window),
                }
                self.tracks.append(tr)
                used.add(len(self.tracks) - 1)
            tr['bbox'] = bbox
            tr['last_seen'] = self.frame_num
            if ok:
                tr['votes'].append(ident.id)
            # Spoof vote runs every frame regardless of quality gate.
            tr['spoof_votes'].append(bool(raw_spoof))

            counts = collections.Counter(v for v in tr['votes'] if v != FaceIdentifier.UNKNOWN_ID)
            if counts:
                top_id, top_count = counts.most_common(1)[0]
            else:
                top_id, top_count = FaceIdentifier.UNKNOWN_ID, 0
            confirmed_id = top_id if top_count >= self.min_votes else FaceIdentifier.UNKNOWN_ID
            smoothed.append(FaceIdentifier.Result(confirmed_id, ident.distance, ident.descriptor))

            spoof_count = sum(tr['spoof_votes'])
            confirmed_spoof = spoof_count >= self.min_votes
            smoothed_spoof.append(confirmed_spoof)

        self.tracks = [t for t in self.tracks if self.frame_num - t['last_seen'] < self.window]
        return smoothed, smoothed_spoof


class FrameProcessor:
    QUEUE_SIZE = 16

    def __init__(self, args):
        self.allow_grow = args.allow_grow and not args.no_show

        log.info('OpenVINO Runtime')
        log.info('\tbuild: {}'.format(get_version()))
        core = Core()

        self.face_detector = FaceDetector(core, args.m_fd,
                                          args.fd_input_size,
                                          confidence_threshold=args.t_fd,
                                          roi_scale_factor=args.exp_r_fd)
        self.landmarks_detector = LandmarksDetector(core, args.m_lm)
        self.face_identifier = FaceIdentifier(core, args.m_reid,
                                              match_threshold=args.t_id,
                                              match_algo=args.match_algo)

        self.face_detector.deploy(args.d_fd)
        self.landmarks_detector.deploy(args.d_lm, self.QUEUE_SIZE)
        self.face_identifier.deploy(args.d_reid, self.QUEUE_SIZE)

        self.anti_spoof = None
        if args.m_as is not None:
            self.anti_spoof = AntiSpoof(core, args.m_as,
                                        scale=args.as_scale,
                                        threshold=args.as_threshold)
            self.anti_spoof.deploy(args.d_as, self.QUEUE_SIZE)

        log.debug('Building faces database using images from {}'.format(args.fg))
        self.faces_database = FacesDatabase(args.fg, self.face_identifier,
                                            self.landmarks_detector,
                                            self.face_detector if args.run_detector else None,
                                            args.no_show,
                                            augment=not args.no_enroll_augment)
        self.face_identifier.set_faces_database(self.faces_database)
        log.info('Database is built, registered {} identities'.format(len(self.faces_database)))

        if self.anti_spoof is not None:
            self._calibrate_anti_spoof_from_gallery(args.fg)

        self.smoother = None
        if args.smooth_window > 0:
            self.smoother = TemporalSmoother(window=args.smooth_window,
                                             min_votes=args.smooth_min_votes)
        self.min_face_size = args.min_face_size
        self.min_blur_var = args.min_blur_var

        self.attendance = None
        if args.db_url:
            self.attendance = AttendanceLogger(args.db_url, args.snapshot_dir)
            log.info('Attendance logging enabled: %s', args.db_url)
        # Most-recent "✓ marked" banner shown on screen.
        self.flash = None  # (name, time_str, expires_at_monotonic)

    def _calibrate_anti_spoof_from_gallery(self, gallery_path):
        """Auto-pick the 'real' class index by running the anti-spoof model on
        every gallery image (guaranteed real faces) and taking the argmax of
        the mean probability vector. Runs once at startup. No user interaction.
        """
        from os import listdir
        from os.path import join, isdir
        accum = np.zeros(3, dtype=np.float64)
        counted = 0
        if not isdir(gallery_path):
            return
        for fname in listdir(gallery_path):
            if fname.split('.')[-1].lower() not in ('jpg', 'jpeg', 'png'):
                continue
            img = cv2.imread(join(gallery_path, fname), cv2.IMREAD_COLOR)
            if img is None:
                continue
            rois = self.face_detector.infer((img,))
            if not rois:
                continue
            rois = [max(rois, key=lambda r: r.size[0] * r.size[1])]
            probs_list = self.anti_spoof.infer_probs(img, rois)
            if probs_list and probs_list[0] is not None and probs_list[0].size == 3:
                accum += probs_list[0]
                counted += 1
        if counted == 0:
            log.warning('Anti-spoof calibration: no faces in gallery; using default real_class=1')
            return
        mean = accum / counted
        self.anti_spoof.real_class = int(np.argmax(mean))
        log.info('Anti-spoof calibrated from gallery: real_class={} (mean probs: {})'
                 .format(self.anti_spoof.real_class, np.array2string(mean, precision=3)))

    def _quality_ok(self, frame, roi):
        w = float(roi.size[0])
        h = float(roi.size[1])
        if self.min_face_size > 0 and min(w, h) < self.min_face_size:
            return False
        if self.min_blur_var > 0:
            x1 = max(int(roi.position[0]), 0)
            y1 = max(int(roi.position[1]), 0)
            x2 = min(int(roi.position[0] + w), frame.shape[1])
            y2 = min(int(roi.position[1] + h), frame.shape[0])
            if x2 <= x1 or y2 <= y1:
                return False
            crop_img = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            if cv2.Laplacian(gray, cv2.CV_64F).var() < self.min_blur_var:
                return False
        return True

    def process(self, frame):
        orig_image = frame.copy()

        rois = self.face_detector.infer((frame,))
        if self.QUEUE_SIZE < len(rois):
            log.warning('Too many faces for processing. Will be processed only {} of {}'
                        .format(self.QUEUE_SIZE, len(rois)))
            rois = rois[:self.QUEUE_SIZE]

        spoof_mask = [False] * len(rois)
        if self.anti_spoof is not None and len(rois) > 0:
            real_scores = self.anti_spoof.infer_scores(orig_image, rois)
            spoof_mask = [s < self.anti_spoof.threshold for s in real_scores]

        landmarks = self.landmarks_detector.infer((frame, rois))
        face_identities, unknowns = self.face_identifier.infer((frame, rois, landmarks))

        for i, is_spoof in enumerate(spoof_mask):
            if is_spoof:
                face_identities[i].id = SPOOF_ID
        if self.allow_grow and len(unknowns) > 0:
            for i in unknowns:
                # This check is preventing asking to save half-images in the boundary of images
                if rois[i].position[0] == 0.0 or rois[i].position[1] == 0.0 or \
                    (rois[i].position[0] + rois[i].size[0] > orig_image.shape[1]) or \
                    (rois[i].position[1] + rois[i].size[1] > orig_image.shape[0]):
                    continue
                crop_image = crop(orig_image, rois[i])
                name = self.faces_database.ask_to_save(crop_image)
                if name:
                    id = self.faces_database.dump_faces(crop_image, face_identities[i].descriptor, name)
                    face_identities[i].id = id

        if self.smoother is not None:
            quality_ok = [self._quality_ok(orig_image, r) and not spoof_mask[i]
                          for i, r in enumerate(rois)]
            face_identities, smoothed_spoof = self.smoother.update(
                rois, face_identities, quality_ok, spoof_mask)
            for i, is_spoof in enumerate(smoothed_spoof):
                if is_spoof:
                    face_identities[i].id = SPOOF_ID
        else:
            smoothed_spoof = spoof_mask
            for i, is_spoof in enumerate(spoof_mask):
                if is_spoof:
                    face_identities[i].id = SPOOF_ID

        if self.attendance is not None:
            import time
            for i, ident in enumerate(face_identities):
                if ident.id == SPOOF_ID:
                    self.attendance.log_spoof(orig_image, rois[i])
                elif ident.id != FaceIdentifier.UNKNOWN_ID and not smoothed_spoof[i]:
                    name = self.face_identifier.get_identity_label(ident.id)
                    confidence = 1.0 - float(ident.distance)
                    marked, time_str = self.attendance.mark(name, confidence, orig_image, rois[i])
                    if marked:
                        self.flash = (name, time_str, time.monotonic() + 2.5)

        return [rois, landmarks, face_identities]


def draw_flash(frame, frame_processor):
    import time
    flash = frame_processor.flash
    if flash is None:
        return
    name, time_str, expires = flash
    if time.monotonic() > expires:
        frame_processor.flash = None
        return
    h, w = frame.shape[:2]
    banner_h = 50
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 180, 0), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    msg = 'Marked: {} at {}'.format(name, time_str)
    cv2.putText(frame, msg, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)


def draw_detections(frame, frame_processor, detections, output_transform):
    size = frame.shape[:2]
    frame = output_transform.resize(frame)
    for roi, landmarks, identity in zip(*detections):
        if identity.id == SPOOF_ID:
            text = 'SPOOF'
            box_color = (0, 0, 220)
        else:
            text = frame_processor.face_identifier.get_identity_label(identity.id)
            box_color = (0, 220, 0)
            if identity.id != FaceIdentifier.UNKNOWN_ID:
                text += ' %.2f%%' % (100.0 * (1 - identity.distance))

        xmin = max(int(roi.position[0]), 0)
        ymin = max(int(roi.position[1]), 0)
        xmax = min(int(roi.position[0] + roi.size[0]), size[1])
        ymax = min(int(roi.position[1] + roi.size[1]), size[0])
        xmin, ymin, xmax, ymax = output_transform.scale([xmin, ymin, xmax, ymax])
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), box_color, 2)

        for point in landmarks:
            x = xmin + output_transform.scale(roi.size[0] * point[0])
            y = ymin + output_transform.scale(roi.size[1] * point[1])
            cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 255), 2)
        textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
        cv2.rectangle(frame, (xmin, ymin), (xmin + textsize[0], ymin - textsize[1]), (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, text, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

    return frame

def center_crop(frame, crop_size):
    fh, fw, _ = frame.shape
    crop_size[0], crop_size[1] = min(fw, crop_size[0]), min(fh, crop_size[1])
    return frame[(fh - crop_size[1]) // 2 : (fh + crop_size[1]) // 2,
                 (fw - crop_size[0]) // 2 : (fw + crop_size[0]) // 2,
                 :]

def main():
    args = build_argparser().parse_args()

    cap = open_images_capture(args.input, args.loop)
    frame_processor = FrameProcessor(args)

    frame_num = 0
    metrics = PerformanceMetrics()
    presenter = None
    output_transform = None
    input_crop = None
    if args.crop_size[0] > 0 and args.crop_size[1] > 0:
        input_crop = np.array(args.crop_size)
    elif not (args.crop_size[0] == 0 and args.crop_size[1] == 0):
        raise ValueError('Both crop height and width should be positive')
    video_writer = cv2.VideoWriter()

    while True:
        start_time = perf_counter()
        frame = cap.read()
        if frame is None:
            if frame_num == 0:
                raise ValueError("Can't read an image from the input")
            break
        if input_crop is not None:
            frame = center_crop(frame, input_crop)
        if frame_num == 0:
            output_transform = OutputTransform(frame.shape[:2], args.output_resolution)
            if args.output_resolution:
                output_resolution = output_transform.new_resolution
            else:
                output_resolution = (frame.shape[1], frame.shape[0])
            presenter = monitors.Presenter(args.utilization_monitors, 55,
                                           (round(output_resolution[0] / 4), round(output_resolution[1] / 8)))
            if args.output and not video_writer.open(args.output, cv2.VideoWriter_fourcc(*'MJPG'),
                                                     cap.fps(), output_resolution):
                raise RuntimeError("Can't open video writer")

        detections = frame_processor.process(frame)
        presenter.drawGraphs(frame)
        frame = draw_detections(frame, frame_processor, detections, output_transform)
        draw_flash(frame, frame_processor)
        metrics.update(start_time, frame)

        frame_num += 1
        if video_writer.isOpened() and (args.output_limit <= 0 or frame_num <= args.output_limit):
            video_writer.write(frame)

        if not args.no_show:
            cv2.imshow('Face recognition demo', frame)
            key = cv2.waitKey(1)
            # Quit
            if key in {ord('q'), ord('Q'), 27}:
                break
            presenter.handleKey(key)

    metrics.log_total()
    for rep in presenter.reportMeans():
        log.info(rep)


if __name__ == '__main__':
    sys.exit(main() or 0)
