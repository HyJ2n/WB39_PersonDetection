# 특정 얼굴 트래킹 일단 성공 , 수정이 좀 더 필요함 

import cv2
import os
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from ultralytics import YOLO
from datetime import datetime, timedelta
from age_model import ResNetAgeModel, device, test_transform
from PIL import Image
from collections import Counter
from deep_sort_realtime.deepsort_tracker import DeepSort

class FaceRecognizer:
    def __init__(self, device=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.mtcnn = MTCNN(keep_all=True, post_process=False, device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.tracked_faces = {}  # 트래킹 ID와 face_id를 매핑할 딕셔너리
        self.known_faces = set()  # 이미 트래킹 중인 face_id를 저장하는 집합
        

    def extract_embeddings(self, image):
        boxes, probs = self.mtcnn.detect(image)
        embeddings = []
        if boxes is not None:
            for box, prob in zip(boxes, probs):
                if prob < 0.99:
                    continue
                box = [int(coord) for coord in box]
                face = image[box[1]:box[3], box[0]:box[2]]
                if face.size == 0:
                    continue
                face_tensor = torch.tensor(face).unsqueeze(0).permute(0, 3, 1, 2).float().to(self.device) / 255.0
                face_resized = torch.nn.functional.interpolate(face_tensor, size=(160, 160), mode='bilinear')
                embedding = self.resnet(face_resized).detach().cpu().numpy().flatten()               
                embeddings.append((embedding, box))
        return embeddings

    def assign_face_id(self, face_encoding, known_faces, threshold=0.53):
        if known_faces:
            similarities = [self.compare_similarity(face_encoding, face['embedding']) for face in known_faces]
            max_similarity = max(similarities)
            max_similarity_index = similarities.index(max_similarity)
            if max_similarity > threshold:
                return known_faces[max_similarity_index]['id']
        return None
    
    def load_known_faces(self, image_paths):
        known_faces = []
        for image_path in image_paths:
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
            embeddings = self.extract_embeddings(image)
            if embeddings:
                embedding, box = embeddings[0]
                face_id = os.path.splitext(os.path.basename(image_path))[0]  # 파일 이름을 ID로 사용
                known_faces.append({'embedding': embedding, 'box': box, 'id': face_id})
        return known_faces

    @staticmethod
    def compare_similarity(embedding1, embedding2):
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        print(f"Similarity 유사도 입니다.: {similarity}")
        return similarity

    def detect_persons(self, frame, yolo_model):
        yolo_results = yolo_model.predict(source=[frame], save=False)[0]
        person_detections = [
            (int(data[0]), int(data[1]), int(data[2]), int(data[3]))
            for data in yolo_results.boxes.data.tolist()
            if float(data[4]) >= 0.85 and int(data[5]) == 0
        ]
        return person_detections

    def draw_bounding_boxes(self, frame, tracks, tracked_faces):
        for track in tracks:
            if not track.is_confirmed():
                continue

            ltrb = track.to_ltrb()
            track_bbox = (int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3]))

            if track.track_id in tracked_faces:
                face_id = tracked_faces[track.track_id]
                id_text = f"face_id: {face_id}"

                cv2.rectangle(frame, (track_bbox[0], track_bbox[1]), (track_bbox[2], track_bbox[3]), (0, 255, 0), 2)
                cv2.putText(frame, id_text, (track_bbox[0], track_bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    def recognize_faces(self, frame, frame_number, output_dir, known_faces, tracker, video_name, yolo_model, gender_model, age_model):
        original_shape = frame.shape[:2]
        frame_resized = cv2.resize(frame, (640, 480))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        detect_faces = self.extract_embeddings(frame_rgb)
        person_detections = self.detect_persons(frame, yolo_model)

        results = []
        for (xmin, ymin, xmax, ymax) in person_detections:
            results.append([[xmin, ymin, xmax-xmin, ymax-ymin], 1.0, 0])

        tracks = tracker.update_tracks(results, frame=frame)

                # 트래킹에서 사라진 얼굴을 집합에서 제거
        active_track_ids = {track.track_id for track in tracks if track.is_confirmed()}
        inactive_track_ids = set(self.tracked_faces.keys()) - active_track_ids
        for inactive_track_id in inactive_track_ids:
            if self.tracked_faces[inactive_track_id] in self.known_faces:
                self.known_faces.remove(self.tracked_faces[inactive_track_id])
            del self.tracked_faces[inactive_track_id]

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            track_bbox = (int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3]))

            for embedding, box in detect_faces:
                left, top, right, bottom = box
                face_center_x = (left + right) / 2
                face_center_y = (top + bottom) / 2

                # 중심 부분을 작은 박스로 설정
                center_box_left = face_center_x - (right - left) / 6
                center_box_top = face_center_y - (bottom - top) / 6
                center_box_right = face_center_x + (right - left) / 6
                center_box_bottom = face_center_y + (bottom - top) / 6
                # if left >= track_bbox[0] and right <= track_bbox[2] and top >= track_bbox[1] and bottom <= track_bbox[3]:
                if (track_bbox[0] <= center_box_left <= track_bbox[2] or track_bbox[0] <= center_box_right <= track_bbox[2]) and \
                   (track_bbox[1] <= center_box_top <= track_bbox[3] or track_bbox[1] <= center_box_bottom <= track_bbox[3]):
                    face_id = self.assign_face_id(embedding, known_faces)
                    if face_id is not None:
                        if face_id in self.known_faces:
                            continue  # 이미 트래킹 중인 얼굴이면 무시
                        self.tracked_faces[track_id] = face_id
                        self.known_faces.add(face_id)
                    break

        self.draw_bounding_boxes(frame, tracks, self.tracked_faces)

        return frame


def process_video(video_path, output_dir, known_face_paths, yolo_model_path, gender_model_path, age_model_path, target_fps=10, max_age=30, n_init=3, nn_budget=60):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    recognizer = FaceRecognizer()

    v_cap = cv2.VideoCapture(video_path)
    if not v_cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frame_rate = int(v_cap.get(cv2.CAP_PROP_FPS))
    frame_interval = frame_rate // target_fps
    frame_width, frame_height = None, None
    video_writer = None

    known_faces = recognizer.load_known_faces(known_face_paths)
    yolo_model = YOLO(yolo_model_path)
    gender_model = YOLO(gender_model_path)
    age_model = ResNetAgeModel(num_classes=4)
    age_model.load_state_dict(torch.load(age_model_path))
    age_model = age_model.to(device)
    age_model.eval()

    tracker = DeepSort(max_age=max_age, n_init=n_init, nn_budget=nn_budget)

    frame_number = 0

    while True:
        success, frame = v_cap.read()
        if not success:
            break
        frame_number += 1

        if frame_number % frame_interval != 0:
            continue

        frame = recognizer.recognize_faces(frame, frame_number, output_dir, known_faces, tracker, video_name, yolo_model, gender_model, age_model)

        if frame_width is None or frame_height is None:
            frame_height, frame_width = frame.shape[:2]
            output_video_path = os.path.join(output_dir, f"{video_name}_output.mp4")
            video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), target_fps, (frame_width, frame_height))

        video_writer.write(frame)

        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    v_cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()

def process_videos(video_paths, output_dir, known_face_paths, yolo_model_path, gender_model_path, age_model_path, target_fps=10):
    for video_path in video_paths:
        process_video(video_path, output_dir, known_face_paths, yolo_model_path, gender_model_path, age_model_path, target_fps)

if __name__ == "__main__":
    video_paths = ["./test/testVideo2.mp4" ,"./test/testVideo.mp4"]
    known_face_paths = [
        "./test/hyojin.png",
        "./test/hyojin2.jpg"
    ]

    output_directory = "./output/"
    yolo_model_path = './models/yolov8x.pt'
    gender_model_path = './models/gender_model.pt'
    age_model_path = './models/age_model.pth'

    process_videos(video_paths, output_directory, known_face_paths, yolo_model_path, gender_model_path, age_model_path)
