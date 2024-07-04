import os
import glob
import pickle
import face_recognition as fr
from tqdm import tqdm
import numpy as np

def extract_faces(train_path):
    face_list = []
    non_recognition_idx = []
    num = 0
    i = 0
    for person in train_path:
        image = fr.load_image_file(person)
        encodings = fr.face_encodings(image)
        if len(encodings) > 0:
            top, right, bottom, left = fr.face_locations(image)[0]
            face_image = image[top:bottom, left:right]
            face_list.append(face_image)
        else:
            non_recognition_idx.append(i)
            num = num + 1
        i = i + 1
        if i % 100 == 0:
            print(i)

    print("얼굴 인식한 이미지 수 : ", len(train_path) - num)
    print(non_recognition_idx)

    for idx in sorted(non_recognition_idx, reverse=True):
        del train_path[idx]

    return train_path, face_list

def preprocess_images(train_path):
    # 캐시 파일 경로 설정
    cache_file = r"C:\Users\admin\Desktop\models\preprocessed_data.pkl"
    
    # 이미 캐시 파일이 있는지 확인하고 로드
    if os.path.exists(cache_file):
        print("Loading preprocessed data from cache...")
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        return data['train_path'], data['face_list']
    
    # 캐시 파일이 없는 경우 이미지 전처리 실행
    print("Preprocessing images...")
    
    # 얼굴 추출
    train_path, face_list = extract_faces(train_path)
    
    # 전처리 결과를 캐시 파일에 저장
    with open(cache_file, 'wb') as f:
        pickle.dump({'train_path': train_path, 'face_list': face_list}, f)
    
    return train_path, face_list

if __name__ == "__main__":
    # 예시로 train_path를 정의하고 사용
    train_path = sorted(glob.glob(r"C:\Users\admin\Desktop\Project\Main\Age\test\All-Age-Faces Dataset\original images\*.jpg"))  # 실제 데이터셋 경로에 맞게 설정
    
    # 이미지 전처리 실행
    train_path, face_list = preprocess_images(train_path)
