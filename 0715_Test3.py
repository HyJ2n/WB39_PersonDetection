# 거리값 유사도로 id매핑

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
        self.persons = []
        self.age_predictions = {}

    def extract_embeddings(self, image):
        boxes, probs = self.mtcnn.detect(image)
        if boxes is not None:
            embeddings = []
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
                embeddings.append((embedding, box, face, prob))
            return embeddings
        else:
            return []

    def detect_persons(self, frame, yolo_model):
        yolo_results = yolo_model.predict(source=[frame], save=False)[0]
        person_detections = [
            (int(data[0]), int(data[1]), int(data[2]), int(data[3]))
            for data in yolo_results.boxes.data.tolist()
            if float(data[4]) >= 0.85 and int(data[5]) == 0
        ]
        return person_detections

    def load_known_faces(self, image_paths):
        known_faces = []
        for image_path in image_paths:
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
            embeddings = self.extract_embeddings(image)
            if embeddings:
                embedding, box, face, prob = embeddings[0]
                known_faces.append({'embedding': embedding, 'box': box, 'face': face, 'prob': prob, 'id': len(known_faces) + 1})
        return known_faces

    @staticmethod
    def compare_similarity(embedding1, embedding2):
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

    def assign_face_id(self, face_encoding, known_faces, threshold=0.7):
        if known_faces:
            distances = [self.compare_similarity(face_encoding, face['embedding']) for face in known_faces]
            max_similarity = max(distances)
            max_similarity_index = distances.index(max_similarity)
            if max_similarity > threshold:
                return known_faces[max_similarity_index]['id']
        return None

    def draw_bounding_boxes(self, frame, tracks, tracked_faces):
        for track in tracks:
            if not track.is_confirmed():
                continue

            ltrb = track.to_ltrb()
            track_bbox = (int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3]))

            display_text = f"Track ID: {track.track_id}"

            if track.track_id in tracked_faces:
                person_id = tracked_faces[track.track_id]["person_id"]
                # gender = tracked_faces[track.track_id]["gender"]
                # age = tracked_faces[track.track_id]["age"]

                id_text = f"Person_id: {person_id}"
                # gender_text = f"Gender: {gender}"
                # age_text = f"Age: {age}"

                cv2.rectangle(frame, (track_bbox[0], track_bbox[1]), (track_bbox[2], track_bbox[3]), (0, 255, 0), 2)
                cv2.putText(frame, id_text, (track_bbox[0], track_bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # cv2.putText(frame, gender_text, (track_bbox[0], track_bbox[1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # cv2.putText(frame, age_text, (track_bbox[0], track_bbox[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (track_bbox[0], track_bbox[1]), (track_bbox[2], track_bbox[3]), (0, 0, 255), 2)
                cv2.putText(frame, display_text, (track_bbox[0], track_bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    def recognize_faces(self, frame, frame_number, output_dir, known_faces, tracker, video_name, yolo_model, gender_model, age_model):
        original_shape = frame.shape[:2]
        frame_resized = cv2.resize(frame, (640, 480))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        detect_faces = self.extract_embeddings(frame_rgb)
        person_detections = self.detect_persons(frame, yolo_model)
        results = []
        tracked_faces = {}

        if detect_faces:
            for embedding, box, face_image, prob in detect_faces:
                person_id = self.assign_face_id(embedding, known_faces)
                if person_id is None:
                    person_id = len(self.persons) + 1
                    self.persons.append({'id': person_id, 'embedding': embedding})
                    self.age_predictions[person_id] = {'frames': [], 'age': None}

                left, top, right, bottom = box
                for (xmin, ymin, xmax, ymax) in person_detections:
                    if left >= xmin and right <= xmax and top >= ymin and bottom <= ymax:
                        face_image = frame[top:bottom, left:right]
                        person_image = frame[ymin:ymax, xmin:xmax]

                        results.append([[xmin, ymin, xmax - xmin, ymax - ymin], 1.0, 0])
                        break

                output_folder = os.path.join(output_dir, f'{video_name}_face')
                os.makedirs(output_folder, exist_ok=True)
                output_path = os.path.join(output_folder, f'person_{person_id}_frame_{frame_number}.jpg')
                cv2.imwrite(output_path, cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
                print(f"Saved image: {output_path}, person ID: {person_id}, detection probability: {prob}")

                scale_x = original_shape[1] / 640
                scale_y = original_shape[0] / 480
                box = [int(coord) for coord in box]
                box[0] = int(box[0] * scale_x)
                box[1] = int(box[1] * scale_y)
                box[2] = int(box[2] * scale_x)
                box[3] = int(box[3] * scale_y)

                print(f"Tracking results: {results}")
                tracks = tracker.update_tracks(results, frame=frame)
                print(f"Tracks: {tracks}")



                # 트래킹된 객체에 대한 바운딩 박스 및 텍스트 추가
                for track in tracks:
                    if not track.is_confirmed():
                        continue

                    track_id = track.track_id
                    ltrb = track.to_ltrb()
                    track_bbox = (int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3]))

                    # 트래킹 객체와 얼굴 인식 객체의 바운딩 박스가 겹치는지 확인
                    if left >= track_bbox[0] and right <= track_bbox[2] and top >= track_bbox[1] and bottom <= track_bbox[3]:
                        tracked_faces[track_id] = {
                            "person_id": person_id
                        }
                        break

            self.draw_bounding_boxes(frame, tracks, tracked_faces)

        return frame


def predict_gender(face_image, gender_model):
    face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    results = gender_model.predict(source=[face_rgb], save=False)
    genders = {0: "Male", 1: "Female"}
    gender_id = results[0].boxes.data[0][5].item()
    return genders.get(gender_id, "Unknown")

def predict_age(face_image, age_model):
    if isinstance(face_image, np.ndarray):
        face_image = Image.fromarray(face_image)

    face_tensor = test_transform(face_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = age_model(face_tensor)
        pred = logit.argmax(dim=1, keepdim=True).cpu().numpy()

    age_group = {0: "Child", 1: "Youth", 2: "Middle", 3: "Old"}
    if pred[0][0] < len(age_group):
        return age_group[pred[0][0]]
    else:
        return "Unknown"

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
    video_paths = ["./test/testVideo.mp4"]
    known_face_paths = [
        "./test/horyun.png"
    ]
    output_directory = "./output/"
    yolo_model_path = './models/yolov8x.pt'
    gender_model_path = './models/gender_model.pt'
    age_model_path = './models/age_model.pth'

    process_videos(video_paths, output_directory, known_face_paths, yolo_model_path, gender_model_path, age_model_path)
