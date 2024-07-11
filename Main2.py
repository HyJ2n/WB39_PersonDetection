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
from collections import defaultdict, Counter
from deep_sort_realtime.deepsort_tracker import DeepSort

class FaceRecognizer:
    def __init__(self, device=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.mtcnn = MTCNN(keep_all=True, post_process=False, device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.persons = []
        self.person_id = 0
        self.age_predictions = defaultdict(lambda: {'frames': [], 'age': None})  # 나이 예측 결과를 저장할 딕셔너리 초기화


    # 예측===================================================================

    def predict_gender(self,face_image, gender_model):
        face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        results = gender_model.predict(source=[face_rgb], save=False)
        genders = {0: "Male", 1: "Female"}
        gender_id = results[0].boxes.data[0][5].item()
        return genders.get(gender_id, "Unknown")


    def predict_age(self,face_image, age_model):
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
        
    # ========================================================================

    def detect_persons(self ,frame, yolo_model):
        yolo_results = yolo_model.predict(source=[frame], save=False)[0]
        person_detections = [
            (int(data[0]), int(data[1]), int(data[2]), int(data[3]))
            for data in yolo_results.boxes.data.tolist()
            if float(data[4]) >= 0.85 and int(data[5]) == 0
        ]
        return person_detections


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
    
    def iou(self, box1, box2):
        # box1과 box2는 [xmin, ymin, xmax, ymax] 형식
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
        iou = interArea / float(box1Area + box2Area - interArea)
        return iou
    

    @staticmethod
    def compare_similarity(embedding1, embedding2):
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2)) 


    def recognize_faces(self,tracker,yolo_model, known_face, frame, frame_number, output_dir, video_name, gender_model, age_model):
        original_shape = frame.shape[:2]
        frame_resized = cv2.resize(frame, (640, 480))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        forward_embeddings = self.extract_embeddings(frame_rgb)
        results = []

        person_detections = self.detect_persons(frame,yolo_model)

        for (xmin, ymin, xmax, ymax) in person_detections:
            width = xmax - xmin
            height = ymax - ymin
            results.append([[xmin, ymin, width, height], 1.0, 0])

        tracks = tracker.update_tracks(results, frame=frame)

        embeddings = [face[0] for face in known_face]
        face_to_track = {}

        if forward_embeddings:
            for embedding, box, face_image, prob in forward_embeddings:
                matched = False
                person_id = None

                # known_face와 비교
                for known_id, known_embedding in enumerate(embeddings):                   
                    if self.compare_similarity(embedding, known_embedding) > 0.6:
                        print("얼굴 검출 됐습니다.")
                        person_id = known_id + 1  # ID는 1부터 시작
                        matched = True
                        break        

                if matched:
                    # Draw bounding box
                    scale_x = original_shape[1] / 640
                    scale_y = original_shape[0] / 480
                    box = [int(coord) for coord in box]
                    left = int(box[0] * scale_x)
                    top = int(box[1] * scale_y)
                    right = int(box[2] * scale_x)
                    bottom = int(box[3] * scale_y)

                # 트래킹된 객체와 매칭
                    for track in tracks:
                        if not track.is_confirmed():
                            continue

                        track_id = track.track_id
                        ltrb = track.to_ltrb()
                        track_bbox = (int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3]))

                    # 얼굴 바운딩 박스가 트랙 바운딩 박스 내에 있는지 확인
                        if person_id not in face_to_track:
                            if left >= track_bbox[0] and right <= track_bbox[2] and top >= track_bbox[1] and bottom <= track_bbox[3]:
                                face_to_track[person_id] = track_id

                        if face_to_track.get(person_id) == track_id:
                            cv2.rectangle(frame, (track_bbox[0], track_bbox[1]), (track_bbox[2], track_bbox[3]), (0, 255, 0), 2)
                            cv2.putText(frame, f"ID: {person_id}", (track_bbox[0], track_bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 트래킹된 객체에 대한 바운딩 박스 및 텍스트 추가
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            track_bbox = (int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3]))


        # 트래킹 객체와 얼굴 인식 객체의 바운딩 박스가 겹치는지 확인
            for person_id, tracked_id in face_to_track.items():
                if tracked_id == track_id:
                    cv2.rectangle(frame, (track_bbox[0], track_bbox[1]), (track_bbox[2], track_bbox[3]), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {person_id}", (track_bbox[0], track_bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)                   
            
                        # Get gender prediction
                    gender = self.predict_gender(face_image, gender_model)

                        # Get age prediction
                    if len(self.age_predictions[person_id]['frames']) < 10:
                        age = self.predict_age(face_image, age_model)
                        self.age_predictions[person_id]['frames'].append(age)
                    else:
                        if self.age_predictions[person_id]['age'] is None:
                            most_common_age = Counter(self.age_predictions[person_id]['frames']).most_common(1)[0][0]
                            self.age_predictions[person_id]['age'] = most_common_age
                        age = self.age_predictions[person_id]['age']

                    # Draw gender text below the bounding box
                    gender_text = f"Gender: {gender}"
                    cv2.putText(frame, gender_text, (box[0], box[3] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    # Draw age text below the bounding box
                    age_text = f"Age: {age}"
                    cv2.putText(frame, age_text, (box[0], box[3] + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    # Draw ID text above the bounding box
                    id_text = f"ID: {person_id}"
                    cv2.putText(frame, id_text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return frame



def process_video(known_face_paths,video_path, output_dir, yolo_model_path, gender_model_path, age_model_path, target_fps=10 ,max_age=30, n_init=3, nn_budget=70):
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
    tracker = DeepSort(max_age=max_age, n_init=n_init, nn_budget=nn_budget)

    frame_number = 0
    known_face = recognizer.load_known_faces(known_face_paths)

    while True:
        success, frame = v_cap.read()
        if not success:
            break
        frame_number += 1

        if frame_number % frame_interval != 0:
            continue
        
        frame = recognizer.recognize_faces(tracker,yolo_model,known_face,frame, frame_number, output_dir, video_name, gender_model, age_model)

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

def process_videos(known_face_paths,video_paths, output_dir, yolo_model_path, gender_model_path, age_model_path, target_fps=10):
    for video_path in video_paths:
        process_video(known_face_paths,video_path, output_dir, yolo_model_path, gender_model_path, age_model_path, target_fps)

if __name__ == "__main__":
    video_paths = ["./test/testVideo1.mp4", "./test/testVideo2.mp4"]
    known_face_paths = [
        "./test/horyun.png"
    ]
    output_directory = "./output/"
    yolo_model_path = './models/yolov8x.pt'
    gender_model_path = './models/gender_model.pt'
    age_model_path = './models/age_model.pth'


    process_videos(known_face_paths,video_paths, output_directory, yolo_model_path, gender_model_path,age_model_path)