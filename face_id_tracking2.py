#얼굴 인식과 YOLO 탐지의 일치 확인:

#얼굴 인식을 통해 얻은 face_id가 YOLO 탐지 결과와 일치하는지 확인해야 합니다.
#일치하는 객체만 추적:

#얼굴 인식된 객체와 YOLO 탐지된 객체의 일치 여부를 확인한 후, 일치하는 객체만 추적하도록 합니다.
#추적된 객체의 face_id 유지:

#처음 얼굴 인식된 객체가 추적되면 이후에도 해당 객체가 유지되도록 해야 합니다.

import cv2
import face_recognition
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def predict_clothes(person_image,clothes_model):
    clothes_class_names = {0: 'dress', 1: 'longsleevetop', 2: 'shortsleevetop', 3: 'vest', 4: 'shorts', 5: 'pants', 6: 'skirt'}
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
            cls_name = clothes_class_names.get(cls_id, "Unknown")
            detected_clothes.append(cls_name)

    return detected_clothes

def predict_color(person_image,color_model):
    color_class_names = {0: 'black', 1: 'white', 2: 'red', 3: 'yellow', 4: 'green', 5: 'blue', 6: 'brown'}
    if isinstance(person_image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(person_image, cv2.COLOR_BGR2RGB))
        # 의류 모델 예측 수행
    color_results = color_model(image)
    # 예측 결과를 clothes_class_names와 매핑
    detected_colors = []
    for colors_result in color_results:
        filtered_colors_results = colors_result.boxes
        for box in filtered_colors_results:
            cls_id = int(box.cls.item())
            cls_name = color_class_names.get(cls_id, "Unknown")
            detected_colors.append(cls_name)

    return detected_colors


def assign_face_id(face_encoding, known_faces, threshold=0.4):
    if known_faces:
        distances = face_recognition.face_distance([encoding for _, encoding in known_faces], face_encoding)
        min_distance = np.min(distances)
        min_distance_index = np.argmin(distances)

        if min_distance < threshold:
            return known_faces[min_distance_index][0]
    return None

def load_known_faces(image_paths):
    known_faces = []
    for image_path in image_paths:
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)
        if face_encodings:
            face_id = os.path.splitext(os.path.basename(image_path))[0]
            known_faces.append((face_id, face_encodings[0]))
    return known_faces

def detect_faces(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    return face_locations, face_encodings

def detect_persons(frame, yolo_model):
    yolo_results = yolo_model.predict(source=[frame], save=False)[0]
    person_detections = [
        (int(data[0]), int(data[1]), int(data[2]), int(data[3]))
        for data in yolo_results.boxes.data.tolist()
        if float(data[4]) >= 0.85 and int(data[5]) == 0
    ]
    return person_detections

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

def process_video(video_path, output_dir,known_face_paths, yolo_model_path, gender_model_path, age_model_path,color_model_path,clothes_model_path, 
                   predict_gender_flag=False,predict_age_flag=False,predict_clothes_flag=False,predict_color_flag=False, conf_threshold=0.6, max_age=30, n_init=3, nn_budget=70):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error("Error: Could not open video.")
            return

        os.makedirs(output_dir, exist_ok=True)

        known_faces = load_known_faces(known_face_paths)
        yolo_model = YOLO(yolo_model_path)
        gender_model = YOLO(gender_model_path)
        age_model = ResNetAgeModel(num_classes=4) 
        age_model.load_state_dict(torch.load(age_model_path))
        age_model = age_model.to(device)
        age_model.eval()
        color_model = YOLO(color_model_path)
        clothes_model = YOLO(clothes_model_path)
        tracker = DeepSort(max_age=max_age, n_init=n_init, nn_budget=nn_budget)

        frame_count = 0
        tracked_faces = {}

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            face_locations, face_encodings = detect_faces(frame)
            person_detections = detect_persons(frame, yolo_model)
            results = []
            
            
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                face_id = assign_face_id(face_encoding, known_faces)
                if face_id is not None:
                    for (xmin, ymin, xmax, ymax) in person_detections:
                        if left >= xmin and right <= xmax and top >= ymin and bottom <= ymax:
                            face_image = frame[top:bottom, left:right]
                            person_image = frame[ymin:ymax, xmin:xmax]
                            clothes = predict_clothes(person_image,clothes_model) if predict_clothes_flag else "Unknown"
                            color = predict_color(person_image,color_model) if predict_color_flag else "Unknown"
                            gender = predict_gender(face_image, gender_model) if predict_gender_flag else "Unknown"
                            age = predict_age(face_image, age_model) if predict_age_flag else "Unknown"
                            results.append([[xmin, ymin, xmax-xmin, ymax-ymin], 1.0, 0])
                            break

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
                        clothes = predict_clothes(person_image,clothes_model) if predict_clothes_flag else "Unknown"
                        gender = predict_gender(face_image, gender_model) if predict_gender_flag else "Unknown"
                        age = predict_age(face_image,age_model) if predict_age_flag else "Unknown"
                        color = predict_color(person_image,color_model) if predict_color_flag else "Unknown"
                        
                        tracked_faces[track_id] = {
                            "face_id": face_id,
                            "gender": gender,
                            "age" : age,
                            "clothes" : clothes,
                            "color" : color
                        }
                        break

            draw_bounding_boxes(frame, tracks, tracked_faces, predict_gender_flag,predict_age_flag,predict_clothes_flag,predict_color_flag)

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
    video_path = r"C:\Users\admin\Desktop\123.mp4"
    output_directory = r"C:\Users\admin\Desktop\tracking_image"
    known_face_paths = [
        r"C:\Users\admin\Desktop\data\testImage\horyun.png"
    ]
    yolo_model_path = r'C:\Users\admin\Desktop\Project\Main\Face_Gender\models\yolov8x.pt'
    gender_model_path = r'C:\Users\admin\Desktop\Project\Main\Face_Gender\models\gender_model.pt'  # 성별 예측 모델 경로
    age_model_path = r'C:\Users\admin\Desktop\Project\Main\Face_Gender\models\age_best.pth'
    color_model_path = r'C:\Users\admin\Desktop\Project\Main\Face_Gender\models\color_model.pt'
    clothes_model_path = r'C:\Users\admin\Desktop\Project\Main\Face_Gender\models\clothes_class.pt'

    process_video(video_path, output_directory, known_face_paths, yolo_model_path, gender_model_path,age_model_path,color_model_path,clothes_model_path ,
                  predict_gender_flag=True,predict_age_flag=True, predict_clothes_flag=True, predict_color_flag=True)
