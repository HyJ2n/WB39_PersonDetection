#GPU
import cv2
from facenet_pytorch import InceptionResnetV1, MTCNN
import numpy as np
import os
import torch
import logging
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
from age_model import ResNetAgeModel, ageDataset, device, test_transform, CFG
from collections import Counter


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def load_image(image_path):
    image = Image.open(image_path)
    image = image.convert('RGB')
    return image



def extract_face_embeddings(image_path, face_model, mtcnn):
    image = load_image(image_path)
    frame_tensor = preprocess_frame(image, device)
    
    # 얼굴 감지 및 추출
    faces = mtcnn(frame_tensor)
    
    if faces is None:
        raise ValueError("No face detected in the image")
    
    embeddings = []
    for face in faces:
        embedding = face_model(face.unsqueeze(0))
        embeddings.append(embedding.detach().cpu().numpy())
    
    return embeddings



def load_known_faces(image_paths, face_model, mtcnn):
    known_faces = []
    for image_path in image_paths:
        try:
            face_encodings = extract_face_embeddings(image_path, face_model, mtcnn)
            face_id = os.path.splitext(os.path.basename(image_path))[0]
            known_faces.append((face_id, face_encodings[0]))  # Assuming only one face is detected and processed
        except ValueError as e:
            print(f"Error processing {image_path}: {e}")
    return known_faces


def calculate_face_distance(face_encoding1, face_encoding2):
    # 두 얼굴 벡터 간의 거리 계산 (유클리드 거리 사용)
    return np.linalg.norm(face_encoding1 - face_encoding2)



def assign_face_id(face_encoding, known_faces, threshold=0.4):
    if known_faces:
        # 주어진 얼굴 벡터와 알려진 얼굴 벡터들 사이의 거리 계산
        distances = [calculate_face_distance(encoding, face_encoding) for _, encoding in known_faces]
        # 가장 짧은 거리와 그 인덱스 찾기
        min_distance = np.min(distances)
        min_distance_index = np.argmin(distances)

        # 최소 거리가 임계값보다 작은 경우 해당 얼굴의 ID 반환
        if min_distance < threshold:
            return known_faces[min_distance_index][0]
    
    # 모든 조건이 만족하지 않으면 None 반환
    return None


def preprocess_frame(frame, device):
    # 이미지 크기를 160x160으로 리사이즈
    resized_frame = frame.resize((160, 160))
    
    # numpy 배열을 PyTorch 텐서로 변환하고 채널 순서를 변경 (H, W, C) -> (C, H, W)
    frame_tensor = torch.tensor(resized_frame).permute(2, 0, 1)
    
    # 배치 차원을 추가하고 float 타입으로 변환
    frame_tensor = frame_tensor.unsqueeze(0).float().to(device)
    
    return frame_tensor



# 얼굴 감지 및 위치 변환 함수
def detect_faces(frame_tensor):

    # 얼굴 감지 및 위치 추출
    face_locations = mtcnn.detect(frame_tensor)

    print(face_locations)

   
    
   

def detect_persons(frame, yolo_model):
    yolo_results = yolo_model.predict(source=[frame], save=False)[0]
    person_detections = [
        (int(data[0]), int(data[1]), int(data[2]), int(data[3]))
        for data in yolo_results.boxes.data.tolist()
        if float(data[4]) >= 0.85 and int(data[5]) == 0
    ]
    return person_detections


def predict_age(face_image, age_model):
    if isinstance(face_image, np.ndarray):
        face_image = Image.fromarray(face_image)

    # 이미지를 Torch 텐서로 변환 및 장치로 이동
    face_tensor = test_transform(face_image).unsqueeze(0).to(device)

    # 예측 수행
    with torch.no_grad():
        logit = age_model(face_tensor)
        pred = logit.argmax(dim=1, keepdim=True).cpu().numpy()

    # 클래스에 따라 나이 그룹 반환
    age_group = ["Child", "Youth", "Middle", "Old"]
    if pred[0][0] < len(age_group):
        return age_group[pred[0][0]]
    else:
        return "Unknown"
    
def predict_gender(face_image, gender_model):
    results = gender_model.predict(source=face_image, device='cuda', conf=0.6)
    if results and len(results) > 0 and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            label = int(box.cls)
            if label == 0:  # Assuming 80 is Male
                return "Male"
            elif label == 1:  # Assuming 81 is Female
                return "Female"
    return "Unknown"

def predict_clothes(person_image, clothes_model):
    clothes_class_names = ['dress', 'longsleevetop', 'shortsleevetop', 'vest', 'shorts', 'pants', 'skirt']

    if isinstance(person_image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(person_image, cv2.COLOR_BGR2RGB))

    # 의류 모델 예측 수행
    clothes_results = clothes_model(image)

    # 예측 결과를 clothes_class_names와 매핑
    detected_clothes = []
    for clothes_result in clothes_results:
        filtered_clothes_results = clothes_result.boxes
        for box in filtered_clothes_results:
            cls_id = int(box.cls.item())
            cls_name = clothes_class_names[cls_id] if 0 <= cls_id < len(clothes_class_names) else "Unknown"
            detected_clothes.append(cls_name)

    return detected_clothes


def predict_color(person_image, color_model):
    color_class_names = ['black', 'white', 'red', 'yellow', 'green', 'blue', 'brown']

    if isinstance(person_image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(person_image, cv2.COLOR_BGR2RGB))

    # 색상 모델 예측 수행
    color_results = color_model(image)

    # 예측 결과를 color_class_names와 매핑
    detected_colors = []
    for color_result in color_results:
        filtered_color_results = color_result.boxes
        for box in filtered_color_results:
            cls_id = int(box.cls.item())
            cls_name = color_class_names[cls_id] if 0 <= cls_id < len(color_class_names) else "Unknown"
            detected_colors.append(cls_name)

    return detected_colors



def draw_bounding_boxes(frame, tracks, tracked_faces, predict_gender_flag, predict_age_flag,predict_clothes_flag,predict_color_flag):
    for track in tracks:
        if not track.is_confirmed():
            continue

        ltrb = track.to_ltrb()
        track_bbox = (int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3]))

        
        display_text = f"Track ID: {track.track_id}"
        
        if track.track_id in tracked_faces:
            face_id = tracked_faces[track.track_id]["face_id"]
            gender = tracked_faces[track.track_id]["gender"]
            age = tracked_faces[track.track_id]["age"]
            clothes = tracked_faces[track.track_id]["clothes"]
            color = tracked_faces[track.track_id]["color"]
            display_text = f"Face ID: {face_id}"
            if predict_gender_flag:
                display_text += f", Gender: {gender}"
            if predict_age_flag:
                display_text += f", Age: {age}"
            if predict_clothes_flag:
                display_text += f", Clothes: {clothes}"
            if predict_color_flag:
                display_text += f", Color: {color}"

            cv2.rectangle(frame, (track_bbox[0], track_bbox[1]), (track_bbox[2], track_bbox[3]), (0, 255, 0), 2)
            cv2.putText(frame, display_text, (track_bbox[0], track_bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        else:
            display_text = f"Track ID: {track.track_id}"
            # 바운딩 박스를 그리지 않음
            # cv2.putText(frame, display_text, (track_bbox[0], track_bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


def process_video(face_model,mtcnn,video_path, output_dir, known_face_paths, yolo_model_path, gender_model_path, age_model_path, color_model_path, clothes_model_path,
                   predict_gender_flag=False, predict_age_flag=False, predict_clothes_flag=False, predict_color_flag=False, conf_threshold=0.6, max_age=30, n_init=3, nn_budget=70):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error("Error: Could not open video.")
            return

        os.makedirs(output_dir, exist_ok=True)
        
        yolo_model = YOLO(yolo_model_path)
        gender_model = YOLO(gender_model_path)
        age_model = ResNetAgeModel(num_classes=4)
        age_model.load_state_dict(torch.load(age_model_path))
        age_model = age_model.to(device)
        age_model.eval()
        color_model = YOLO(color_model_path)
        clothes_model = YOLO(clothes_model_path)
        tracker = DeepSort(max_age=max_age, n_init=n_init, nn_budget=nn_budget)
        known_faces = load_known_faces(known_face_paths, face_model, mtcnn)

        frame_count = 0
        tracked_faces = {}
        age_predictions = []
        gender_predictions = []
        color_predictions = []
        clothes_predictions = []

        most_common_age = "Unknown"
        most_common_gender = "Unknown"
        most_common_color = "Unknown"
        most_common_clothes = "Unknown"

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            
            # face_locations, face_encodings = 
            frame_tensor = preprocess_frame(frame, device)
            
            face_locations, face_encodings = detect_faces(frame_tensor)    
            person_detections = detect_persons(frame, yolo_model)
            results = []

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                face_id = assign_face_id(face_encoding, known_faces)
                if face_id is not None:
                    for (xmin, ymin, xmax, ymax) in person_detections:
                        if left >= xmin and right <= xmax and top >= ymin and bottom <= ymax:
                            face_image = frame[top:bottom, left:right]
                            person_image = frame[ymin:ymax, xmin:xmax]
                            clothes = predict_clothes(person_image, clothes_model) if predict_clothes_flag else "Unknown"
                            clothes_predictions.append(clothes)

                            color = predict_color(person_image, color_model) if predict_color_flag else "Unknown"
                            color_predictions.append(color)

                            gender = predict_gender(face_image, gender_model) if predict_gender_flag else "Unknown"
                            gender_predictions.append(gender)

                            age = predict_age(face_image, age_model) if predict_age_flag else "Unknown"
                            age_predictions.append(age)

                            results.append([[xmin, ymin, xmax - xmin, ymax - ymin], 1.0, 0])
                            break

            # 첫 번째 프레임 이후에는 가장 빈도가 높은 나이 값을 업데이트
            if frame_count > 0 and frame_count % 10 == 0 and age_predictions:
                most_common_age = Counter(age_predictions).most_common(1)[0][0]
                age_predictions = []  # 초기화

            if frame_count > 0 and frame_count % 10 == 0 and gender_predictions:
                most_common_gender = Counter(gender_predictions).most_common(1)[0][0]
                gender_predictions = []  

            if frame_count > 0 and frame_count % 10 == 0 and color_predictions:
                color_list = [color for colors in color_predictions for color in colors]
                most_common_color = Counter(color_list).most_common(1)[0][0]
                color_predictions = [] 
            
            if frame_count > 0 and frame_count % 10 == 0 and clothes_predictions:
                clothes_list = [clothes for clotheses in clothes_predictions for clothes in clotheses]
                most_common_clothes = Counter(clothes_list).most_common(1)[0][0]
                clothes_predictions = [] 

            tracks = tracker.update_tracks(results, frame=frame)

            # 트래킹된 객체에 대한 바운딩 박스 및 텍스트 추가
            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                ltrb = track.to_ltrb()
                track_bbox = (int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3]))

                # 트래킹 객체와 얼굴 인식 객체의 바운딩 박스가 겹치는지 확인
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    face_id = assign_face_id(face_encoding, known_faces)
                    if face_id is not None and left >= track_bbox[0] and right <= track_bbox[2] and top >= track_bbox[1] and bottom <= track_bbox[3]:
                        face_image = frame[top:bottom, left:right]
                        person_image = frame[ymin:ymax, xmin:xmax]
                        clothes = most_common_clothes if predict_clothes_flag else "Unknown"
                        gender = most_common_gender if predict_gender_flag else "Unknown"
                        age = most_common_age if predict_age_flag else "Unknown"
                        color = most_common_color if predict_color_flag else "Unknown"

                        tracked_faces[track_id] = {
                            "face_id": face_id,
                            "gender": gender,
                            "age": age,
                            "clothes": clothes,
                            "color": color
                        }
                        break

            draw_bounding_boxes(frame, tracks, tracked_faces, predict_gender_flag, predict_age_flag, predict_clothes_flag, predict_color_flag)

            cv2.imshow('frame', frame)
            output_path = os.path.join(output_dir, f"frame_{frame_count + 1}.jpg")
            cv2.imwrite(output_path, frame)
            logging.info(f"Saved frame {frame_count + 1} with bounding boxes to {output_path}")

            frame_count += 1

            if cv2.waitKey(1) == ord('q'):
                break

        logging.info("Processing complete.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = "./test/testVideo.mp4"
    output_directory = "./output/image1"
    known_face_paths = [
        "./test/horyun.jpg"
    ]
    yolo_model_path = './models/yolov8x.pt'
    gender_model_path = './models/gender_model.pt'  # 성별 예측 모델 경로
    age_model_path = './models/age_best.pth'
    color_model_path = './models/color_model.pt'
    clothes_model_path = './models/clothes_class.pt'
    face_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    mtcnn = MTCNN(keep_all=True, device=device)

    process_video(face_model,mtcnn,video_path, output_directory, known_face_paths, yolo_model_path, gender_model_path,age_model_path,color_model_path,clothes_model_path ,
                  predict_gender_flag=True,predict_age_flag=True, predict_clothes_flag=True, predict_color_flag=True)
