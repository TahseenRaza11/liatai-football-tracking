import cv2
import os
import numpy as np
import pickle
from ultralytics import YOLO
import supervision as sv
from utils import get_center_of_bbox, get_bbox_width

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i: i + batch_size], conf=0.1)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        detections = self.detect_frames(frames)
        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}
            det_sup = sv.Detections.from_ultralytics(detection)
            det_tracked = self.tracker.update_with_detections(det_sup)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for det in det_tracked:
                bbox = det[0].tolist()
                cls_id = det[3]
                track_id = det[4]

                cls_name = cls_names.get(cls_id, "").lower()
                if "player" in cls_name:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                elif "referee" in cls_name or "refree" in cls_name:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            for det in det_sup:
                bbox = det[0].tolist()
                cls_id = det[3]
                cls_name = cls_names.get(cls_id, "").lower()
                if "ball" in cls_name:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        if stub_path:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(frame, (x_center, y2), (int(width), int(0.35 * width)),
                    0.0, -45, 235, color, 2, cv2.LINE_4)

        if track_id is not None:
            rect_w, rect_h = 40, 20
            x1 = x_center - rect_w // 2
            y1 = y2 + 15 - rect_h // 2
            x2 = x_center + rect_w // 2
            y2 = y2 + 15 + rect_h // 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FILLED)
            x_text = x1 + 12 - (10 if track_id > 99 else 0)
            cv2.putText(frame, f"{track_id}", (x_text, y1 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return frame

    def draw_traingle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)
        points = np.array([[x, y], [x - 10, y - 20], [x + 10, y - 20]])
        cv2.drawContours(frame, [points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [points], 0, (0, 0, 0), 2)
        return frame

    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []
        for i, frame in enumerate(video_frames):
            frame = frame.copy()
            for tid, player in tracks["players"][i].items():
                frame = self.draw_ellipse(frame, player["bbox"], (0, 0, 255), tid)
            for _, referee in tracks["referees"][i].items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))
            for _, ball in tracks["ball"][i].items():
                frame = self.draw_traingle(frame, ball["bbox"], (0, 255, 0))
            output_video_frames.append(frame)
        return output_video_frames
