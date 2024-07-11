import cv2
import os
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from ultralytics import YOLO
from age_model import ResNetAgeModel, device, test_transform
from PIL import Image
from collections import Counter
import subprocess

class FaceRecognizer:
    def __init__(self, device=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.mtcnn = MTCNN(keep_all=True, post_process=False, device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.persons = []
        self.person_id = 0
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

    @staticmethod
    def compare_similarity(embedding1, embedding2):
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

    def recognize_faces(self, frame, frame_number, output_dir, video_name, gender_model, age_model):
        original_shape = frame.shape[:2]
        frame_resized = cv2.resize(frame, (640, 480))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        forward_embeddings = self.extract_embeddings(frame_rgb)

        if forward_embeddings:
            for embedding, box, face_image, prob in forward_embeddings:
                matched = False
                for person in self.persons:
                    if self.compare_similarity(embedding, person['embedding']) > 0.7:
                        person_id = person['id']
                        matched = True
                        break
                if not matched:
                    person_id = len(self.persons) + 1
                    self.persons.append({'id': person_id, 'embedding': embedding})
                    self.age_predictions[person_id] = {'frames': [], 'age': None}

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
                output_path = os.path.join(output_folder, f'person_{person_id}_frame_{frame_number}_gender_{gender}_age_{age}.jpg')
                cv2.imwrite(output_path, cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
                print(f"Saved image: {output_path}, person ID: {person_id}, detection probability: {prob}")

                # Draw bounding box
                scale_x = original_shape[1] / 640
                scale_y = original_shape[0] / 480
                box = [int(coord) for coord in box]
                box[0] = int(box[0] * scale_x)
                box[1] = int(box[1] * scale_y)
                box[2] = int(box[2] * scale_x)
                box[3] = int(box[3] * scale_y)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)

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

def process_video(video_path, output_dir, yolo_model_path, gender_model_path, age_model_path, target_fps=10, global_persons=[]):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    recognizer = FaceRecognizer()
    recognizer.persons = global_persons

    v_cap = cv2.VideoCapture(video_path)
    frame_rate = int(v_cap.get(cv2.CAP_PROP_FPS))
    frame_interval = frame_rate // target_fps
    frame_width, frame_height = None, None
    video_writer = None

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

    return recognizer.persons

def process_videos(video_paths, output_dir, yolo_model_path, gender_model_path, age_model_path, target_fps=10):
    global_persons = []
    for video_path in video_paths:
        global_persons = process_video(video_path, output_dir, yolo_model_path, gender_model_path, age_model_path, target_fps, global_persons)

if __name__ == "__main__":
    video_paths = ["./test/testVideo1.mp4", "./test/testVideo2.mp4"]
    output_directory = "./output/"
    yolo_model_path = './models/yolov8x.pt'
    gender_model_path = './models/gender_model.pt'
    age_model_path = './models/age_best.pth'

    process_videos(video_paths, output_directory, yolo_model_path, gender_model_path, age_model_path)
    # process_videos 함수 실행 후 save_face_info3.py 실행
    subprocess.run(["python", "save_face_info3.py"])
    subprocess.run(["python", "make_video_for_face_info3.py"])
    subprocess.run(["python", "extract_best_face2.py"])
