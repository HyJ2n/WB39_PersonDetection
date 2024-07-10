import cv2
import os
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from ultralytics import YOLO
from datetime import datetime, timedelta
from age_model import ResNetAgeModel, ageDataset, device, test_transform, CFG
from PIL import Image

class FaceRecognizer:
    def __init__(self, device=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.mtcnn = MTCNN(keep_all=True, post_process=False, device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.persons = []
        self.person_id = 0

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

    @staticmethod
    def compare_similarity(embedding1, embedding2):
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

    def recognize_faces(self, frame, frame_number, output_dir, video_name, gender_model,age_model):
        original_shape = frame.shape[:2]
        frame_resized = cv2.resize(frame, (640, 480))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        forward_embeddings = self.extract_embeddings(frame_rgb)

        if forward_embeddings:
            for embedding, box, face_image, prob in forward_embeddings:
                matched = False
                for person in self.persons:
                    if self.compare_similarity(embedding, person['embedding']) > 0.7:
                        self.person_id = person['id']
                        matched = True
                        break
                if not matched:
                    self.person_id = len(self.persons) + 1
                    self.persons.append({'id': self.person_id, 'embedding': embedding})
                output_folder = os.path.join(output_dir, f'{video_name}_face')
                os.makedirs(output_folder, exist_ok=True)
                output_path = os.path.join(output_folder, f'person_{self.person_id}_frame_{frame_number}.jpg')
                cv2.imwrite(output_path, cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
                print(f"Saved image: {output_path}, person ID: {self.person_id}, detection probability: {prob}")

                # Draw bounding box
                scale_x = original_shape[1] / 640
                scale_y = original_shape[0] / 480
                box = [int(coord) for coord in box]
                box[0] = int(box[0] * scale_x)
                box[1] = int(box[1] * scale_y)
                box[2] = int(box[2] * scale_x)
                box[3] = int(box[3] * scale_y)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)

                # Get gender prediction
                gender = predict_gender(face_image, gender_model)

                age = predict_age(face_image,age_model)

                # Draw gender text below the bounding box
                gender_text = f" Gender: {gender}"
                cv2.putText(frame, gender_text, (box[0], box[3] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                age_text = f" Age: {age}"
                cv2.putText(frame, age_text, (box[0], box[3] + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Draw ID text above the bounding box
                id_text = f"ID: {self.person_id}"
                cv2.putText(frame, id_text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return frame


def predict_gender(face_image, gender_model):
    face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    results = gender_model.predict(source=[face_rgb], save=False)
    genders = {0: "Male", 1: "Female"}
    gender_id = results[0].boxes.data[0][5].item()
    return genders.get(gender_id, "Unknown")


def predict_age(face_image, age_model):
     # face_image가 numpy 배열이면 PIL 이미지로 변환
    if isinstance(face_image, np.ndarray):
        face_image = Image.fromarray(face_image)

    face_tensor = test_transform(face_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = age_model(face_tensor)
        pred = logit.argmax(dim=1, keepdim=True).cpu().numpy()

    # 클래스에 따라 나이 그룹 반환
    age_group = {0:"Child", 1:"Youth",2: "Middle",3: "Old"}
    if pred[0][0] < len(age_group):
        return age_group[pred[0][0]]
    else:
        return "Unknown"


def process_video(video_path, output_dir, yolo_model_path, gender_model_path,age_model_path, target_fps=10):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    recognizer = FaceRecognizer()

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

        frame = recognizer.recognize_faces(frame, frame_number, output_dir, video_name, gender_model,age_model)

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


def process_videos(video_paths, output_dir, yolo_model_path, gender_model_path,age_model_path, target_fps=10):
    for video_path in video_paths:
        process_video(video_path, output_dir, yolo_model_path, gender_model_path,age_model_path, target_fps)

def get_video_start_time(index):
    start_times = [
        datetime(2024, 7, 4, 10, 2, 0),
        datetime(2024, 7, 4, 10, 1, 0),
        datetime(2024, 7, 4, 10, 0, 30)
    ]
    return start_times[index]


if __name__ == "__main__":
    video_paths = ["./test/testVideo1.mp4", "./test/testVideo2.mp4"]
    output_directory = "./output/"
    yolo_model_path = './models/yolov8x.pt'
    gender_model_path = './models/gender_model.pt'
    age_model_path = './models/age_model.pth'


    process_videos(video_paths, output_directory, yolo_model_path, gender_model_path,age_model_path)