import cv2
import os
import sys
import torch
import numpy as np
from datetime import datetime
from facenet_pytorch import MTCNN, InceptionResnetV1
from ultralytics import YOLO
from age_model import ResNetAgeModel, device, test_transform
from PIL import Image
from collections import Counter

class FaceRecognizer:
    def __init__(self, device=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.mtcnn = MTCNN(keep_all=True, post_process=False, device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.persons = {}
        self.age_predictions = {}

    def extract_embeddings(self, image):
        boxes, probs = self.mtcnn.detect(image)
        if boxes is not None:
            embeddings = []
            for box, prob in zip(boxes, probs):
                if prob < 0.99:
                    continue
                box = [int(coord) for coord in box]
                box = self._expand_box(box, image.shape[:2])
                face = image[box[1]:box[3], box[0]:box[2]]
                if face.size == 0:
                    continue
                # 얼굴 크기가 60x60보다 작은 경우 건너뜁니다.
                if face.shape[0] < 60 or face.shape[1] < 60:
                    continue
                face_tensor = torch.tensor(face).unsqueeze(0).permute(0, 3, 1, 2).float().to(self.device) / 255.0
                face_resized = torch.nn.functional.interpolate(face_tensor, size=(160, 160), mode='bilinear')
                face_image = face_resized.squeeze(0).permute(1, 2, 0).cpu().numpy()
                face_image = (face_image * 255).astype(np.uint8)  # Convert to uint8
                embedding = self.resnet(face_resized).detach().cpu().numpy().flatten()
                embeddings.append((embedding, box, face_image, prob))
            return embeddings
        else:
            return []

    def _expand_box(self, box, image_shape, expand_factor=1.2):
        # 확장된 박스 좌표 계산
        center_x, center_y = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
        width, height = box[2] - box[0], box[3] - box[1]
        new_width, new_height = int(width * expand_factor), int(height * expand_factor)
        
        # 새로운 박스 좌표 계산
        new_x1 = max(center_x - new_width // 2, 0)
        new_y1 = max(center_y - new_height // 2, 0)
        new_x2 = min(center_x + new_width // 2, image_shape[1])
        new_y2 = min(center_y + new_height // 2, image_shape[0])
        
        return [new_x1, new_y1, new_x2, new_y2]

    @staticmethod
    def compare_similarity(embedding1, embedding2):
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

    def recognize_faces(self, frame, frame_number, output_dir, video_name, gender_model, age_model):
        original_shape = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        forward_embeddings = self.extract_embeddings(frame_rgb)

        if forward_embeddings:
            for embedding, box, face_image, prob in forward_embeddings:
                matched = False
                highest_similarity = 0
                best_match_id = None
                
                for person_id, data in self.persons.items():
                    for saved_embedding in data['embeddings']:
                        similarity = self.compare_similarity(embedding, saved_embedding)
                        if similarity > highest_similarity:
                            highest_similarity = similarity
                            best_match_id = person_id
                
                if highest_similarity > 0.6:  # 유사도 임계값
                    person_id = best_match_id
                    matched = True
                else:
                    person_id = len(self.persons) + 1
                    self.persons[person_id] = {'embeddings': []}
                    self.age_predictions[person_id] = {'frames': [], 'age': None}

                self.persons[person_id]['embeddings'].append(embedding)

                # Get gender prediction
                gender = predict_gender(face_image, gender_model)

                # Get age prediction
                if person_id not in self.age_predictions:
                    self.age_predictions[person_id] = {'frames': [], 'age': None}
                
                if len(self.age_predictions[person_id]['frames']) < 10:
                    age = predict_age(face_image, age_model)
                    self.age_predictions[person_id]['frames'].append(age)
                else:
                    if self.age_predictions[person_id]['age'] is None:
                        most_common_age = Counter(self.age_predictions[person_id]['frames']).most_common(1)[0][0]
                        self.age_predictions[person_id]['age'] = most_common_age
                    age = self.age_predictions[person_id]['age']

                output_folder = os.path.join(output_dir, f'{video_name}_face')
                os.makedirs(output_folder, exist_ok=True)
                # 얼굴 이미지를 160x160으로 리사이즈
                face_image_resized = cv2.resize(face_image, (160, 160))
                output_path = os.path.join(output_folder, f'person_{person_id}_frame_{frame_number}_gender_{gender}_age_{age}.jpg')
                cv2.imwrite(output_path, cv2.cvtColor(face_image_resized, cv2.COLOR_RGB2BGR))
                print(f"Saved image: {output_path}, person ID: {person_id}, detection probability: {prob}")

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

def process_video(video_path, output_dir, yolo_model_path, gender_model_path, age_model_path, global_persons={}):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    recognizer = FaceRecognizer()
    recognizer.persons = global_persons

    v_cap = cv2.VideoCapture(video_path)
    frame_rate = int(v_cap.get(cv2.CAP_PROP_FPS))
    frame_interval = 6  # 8프레임마다 처리

    yolo_model = YOLO(yolo_model_path)
    gender_model = YOLO(gender_model_path)
    age_model = ResNetAgeModel(num_classes=4)
    age_model.load_state_dict(torch.load(age_model_path))
    age_model = age_model.to(device)
    age_model.eval()

    frame_number = 0

    while True:
        success, frame = v_cap.read()
        if not success:
            break
        frame_number += 1

        if frame_number % frame_interval != 0:
            continue

        frame = recognizer.recognize_faces(frame, frame_number, output_dir, video_name, gender_model, age_model)

        #cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    v_cap.release()
    cv2.destroyAllWindows()

    return recognizer.persons

def process_videos(video_paths, output_dir, yolo_model_path, gender_model_path, age_model_path):
    global_persons = {}
    for video_path in video_paths:
        global_persons = process_video(video_path, output_dir, yolo_model_path, gender_model_path, age_model_path, global_persons)

if __name__ == "__main__":
    try:
        user_no = sys.argv[2]
        video_directory = f"./uploaded_videos/{user_no}/"
        video_paths = [os.path.join(video_directory, file) for file in os.listdir(video_directory) if file.endswith(('.mp4', '.avi', '.mov'))]
        output_directory = f"./extracted_images/{user_no}/"
        yolo_model_path = './models/yolov8x.pt'
        gender_model_path = './models/gender_model.pt'
        age_model_path = './models/age_best.pth'
        
        process_videos(video_paths, output_directory, yolo_model_path, gender_model_path, age_model_path)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
