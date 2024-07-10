#많이 나온 age와 gender 값으로 고정
import os
import re
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
    
    for file in os.listdir(directory):
        if file.endswith('.jpg'):
            parsed_info = parse_filename(file)
            if parsed_info:
                person_id, frame, gender, age = parsed_info
                if person_id not in info_dict:
                    info_dict[person_id] = []
                info_dict[person_id].append((frame, gender, age))
    
    return info_dict

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
            
            f.write(f'Person ID: {person_id}\n')
            f.write(f'  Gender: {most_common_gender}\n')
            f.write(f'  Age: {most_common_age}\n')
            for frame in sorted(frames):
                f.write(f'  Frame: {frame}\n')
            f.write('\n')

if __name__ == "__main__":
    output_directory = "./output/"
    for video_folder in os.listdir(output_directory):
        folder_path = os.path.join(output_directory, video_folder)
        if os.path.isdir(folder_path):
            info_dict = gather_info_from_files(folder_path)
            output_file = os.path.join(output_directory, f"{video_folder}_info.txt")
            save_info_to_txt(info_dict, output_file)
            print(f"Information saved to {output_file}")
