# 각 인물 고유의 ID 붙이기까지 성공 


import cv2
import os
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from ultralytics import YOLO
from datetime import datetime, timedelta
from age_model import ResNetAgeModel, ageDataset, device, test_transform, CFG
from PIL import Image
from collections import Counter
from deep_sort_realtime.deepsort_tracker import DeepSort

class FaceRecognizer:
    def __init__(self, device=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.mtcnn = MTCNN(keep_all=True, post_process=False, device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.persons = []
        self.person_id = 0
        self.age_predictions = {}  # 나이 예측 결과를 저장할 딕셔너리 초기화

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
            return []  # 얼굴을 감지하지 못한 경우 빈 리스트 반환

    def detect_persons(self ,frame, yolo_model):
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
            # 이미지 로드
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
            
            # 얼굴 임베딩 추출
            embeddings = self.extract_embeddings(image)
            
            if embeddings:
                # 여러 얼굴이 감지될 경우 첫 번째 얼굴만 사용
                embedding, box, face, prob = embeddings[0]
                known_faces.append((embedding, box, face, prob))
        
        return known_faces
    
    

    @staticmethod
    def compare_similarity(embedding1, embedding2):
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    
    def assign_face_id(self, face_encoding, known_faces, threshold=0.7):
        if known_faces:
            # 알려진 얼굴들 중에서 인코딩만 추출하여 리스트 생성
            known_encodings = [encoding for encoding in known_faces]
            # 인식된 얼굴과 알려진 얼굴들 사이의 유사도 계산
            distances = [self.compare_similarity(face_encoding, known_encoding['embedding']) for known_encoding in known_encodings]
            # 계산된 유사도 중 최대값 찾기 (유사도가 가장 높은 것)
            max_similarity = max(distances)
            # 최대값의 인덱스 찾기
            max_similarity_index = distances.index(max_similarity)
            # 최대 유사도가 임계값보다 큰 경우
            if max_similarity > threshold:
                # 해당 인덱스의 알려진 얼굴 ID 반환
                return known_faces[max_similarity_index]['id']

        # 알려진 얼굴이 없거나 임계값보다 큰 유사도가 없는 경우 None 반환
        return None

    def recognize_faces(self, frame, frame_number, output_dir,known_face,tracker, video_name, yolo_model, gender_model, age_model):
        original_shape = frame.shape[:2]
        frame_resized = cv2.resize(frame, (640, 480))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        detect_faces = self.extract_embeddings(frame_rgb)

        person_detections = self.detect_persons(frame, yolo_model)
        
        result =[]

        if detect_faces:
            for embedding, box, face_image, prob in detect_faces:  # 영상에서 검출된 얼굴 위치 , 엔코딩값 전부 가져옴
                person_id = self.assign_face_id(embedding, self.persons)
                if person_id is None:
                    person_id = len(self.persons) + 1
                    self.persons.append({'id': person_id, 'embedding': embedding})
                    self.age_predictions[person_id] = {'frames': [], 'age': None}  # 새로운 사람을 인식할 때 나이 예측 결과 초기화

                output_folder = os.path.join(output_dir, f'{video_name}_face')
                os.makedirs(output_folder, exist_ok=True)
                output_path = os.path.join(output_folder, f'person_{person_id}_frame_{frame_number}.jpg')
                cv2.imwrite(output_path, cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
                print(f"Saved image: {output_path}, person ID: {person_id}, detection probability: {prob}")

                # Draw bounding box for face detection
                scale_x = original_shape[1] / 640
                scale_y = original_shape[0] / 480
                box = [int(coord) for coord in box]
                box[0] = int(box[0] * scale_x)
                box[1] = int(box[1] * scale_y)
                box[2] = int(box[2] * scale_x)
                box[3] = int(box[3] * scale_y)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)  # Red box for face detection

                # Get gender prediction
                gender = predict_gender(face_image, gender_model)

                # Get age prediction
                if len(self.age_predictions[person_id]['frames']) < 10:
                    age = predict_age(face_image, age_model)
                    self.age_predictions[person_id]['frames'].append(age)
                else:
                    if self.age_predictions[person_id]['age'] is None:
                        most_common_age = Counter(self.age_predictions[person_id]['frames']).most_common(1)[0][0]
                        self.age_predictions[person_id]['age'] = most_common_age
                    age = self.age_predictions[person_id]['age']
                    
                # Draw ID text above the bounding box
                id_text = f"ID: {person_id}"
                #cv2.putText(frame, id_text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Draw gender text below the bounding box
                gender_text = f"Gender: {gender}"
                #cv2.putText(frame, gender_text, (box[0], box[3] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Draw age text below the bounding box
                age_text = f"Age: {age}"
                #cv2.putText(frame, age_text, (box[0], box[3] + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Draw ID text above the bounding box
                id_text = f"ID: {person_id}"
                #cv2.putText(frame, id_text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

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

def process_video(video_path, output_dir,known_face_paths, yolo_model_path, gender_model_path, age_model_path, target_fps=10,max_age=30, n_init=3, nn_budget=60):
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

    known_face = recognizer.load_known_faces(known_face_paths)
    known_face_embedding = [face[0] for face in known_face]
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

        frame = recognizer.recognize_faces(frame, frame_number, output_dir,known_face_embedding,tracker, video_name, yolo_model, gender_model, age_model)

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

def process_videos(video_paths, output_dir,known_face_paths, yolo_model_path, gender_model_path, age_model_path, target_fps=10):
    for video_path in video_paths:
        process_video(video_path, output_dir,known_face_paths, yolo_model_path, gender_model_path, age_model_path, target_fps)

if __name__ == "__main__":
    video_paths = ["./test/testVideo1.mp4"] #, "./test/testVideo2.mp4"]
    known_face_paths = [
        "./test/horyun.png"
    ]
    output_directory = "./output/"
    yolo_model_path = './models/yolov8x.pt'
    gender_model_path = './models/gender_model.pt'
    age_model_path = './models/age_model.pth'

    process_videos(video_paths, output_directory,known_face_paths, yolo_model_path, gender_model_path, age_model_path)
