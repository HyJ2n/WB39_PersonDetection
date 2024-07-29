import sys
import os
import re
import shutil
import cv2
from collections import Counter

def parse_filename(filename):
    match = re.match(r'person_(\d+)_frame_(\d+)_gender_(\w+)_age_(\w+)\.jpg', filename)
    if match:
        person_id = int(match.group(1))
        frame = int(match.group(2))
        gender = match.group(3)
        age = match.group(4)
        return person_id, frame, gender, age
    return None

def gather_info_from_files(directory):
    info_dict = {}
    image_files = []

    for file in os.listdir(directory):
        if file.endswith('.jpg'):
            parsed_info = parse_filename(file)
            if parsed_info:
                person_id, frame, gender, age = parsed_info
                if person_id not in info_dict:
                    info_dict[person_id] = []
                info_dict[person_id].append((frame, gender, age))
                image_files.append((person_id, os.path.join(directory, file)))

    for person_id in info_dict:
        info_dict[person_id].sort()  # 프레임을 낮은 순으로 정렬
    
    return info_dict, image_files

def get_most_common(values):
    counter = Counter(values)
    most_common = counter.most_common(1)
    return most_common[0][0] if most_common else None

def save_info_to_txt(info_dict, output_file):
    with open(output_file, 'w') as f:
        for person_id in sorted(info_dict.keys()):
            frames = [frame for frame, gender, age in info_dict[person_id]]
            genders = [gender for frame, gender, age in info_dict[person_id]]
            ages = [age for frame, gender, age in info_dict[person_id]]
            most_common_gender = get_most_common(genders)
            most_common_age = get_most_common(ages)
            
            f.write(f'person_{person_id}:\n')
            f.write(f'  gender: {most_common_gender}\n')
            f.write(f'  age: {most_common_age}\n')
            f.write(f'  frames: {frames}\n\n')

def compute_image_quality(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score

def save_best_faces(image_files, output_folder):
    person_images = {}

    for person_id, filepath in image_files:
        try:
            image = cv2.imread(filepath)
            quality_score = compute_image_quality(image)

            if person_id not in person_images:
                person_images[person_id] = (filepath, quality_score)
            else:
                if quality_score > person_images[person_id][1]:
                    person_images[person_id] = (filepath, quality_score)
        except Exception as e:
            print(f"Error processing file {filepath}: {e}")

    for person_id, (best_image_path, _) in person_images.items():
        person_folder = os.path.join(output_folder, f"person_{person_id}")
        os.makedirs(person_folder, exist_ok=True)
        output_filepath = os.path.join(person_folder, os.path.basename(best_image_path))
        shutil.copy(best_image_path, output_filepath)
        print(f"Copied best quality face image: {output_filepath}")

if __name__ == "__main__":
    try:
        user_id = sys.argv[2]
        #user_id = "1442"
        output_directory = f"./extracted_images/{user_id}/"
        os.makedirs(output_directory, exist_ok=True)  # 디렉토리를 미리 생성합니다.

        for video_folder in os.listdir(output_directory):
            if '_face' in video_folder:
                folder_path = os.path.join(output_directory, video_folder)
                if os.path.isdir(folder_path):
                    try:
                        info_dict, image_files = gather_info_from_files(folder_path)
                        output_file = os.path.join(output_directory, f"{video_folder}_info.txt")
                        save_info_to_txt(info_dict, output_file)
                        print(f"Information saved to {output_file}")

                        clip_folder = folder_path.replace('_face', '_clip')
                        save_best_faces(image_files, clip_folder)
                    except Exception as e:
                        print(f"Error processing folder {folder_path}: {e}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
