import cv2
import os

def extract_frames_from_video(video_path, frames_to_extract):
    v_cap = cv2.VideoCapture(video_path)
    frame_width = int(v_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(v_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(v_cap.get(cv2.CAP_PROP_FPS))

    frames = []
    for frame_number in frames_to_extract:
        v_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = v_cap.read()
        if ret:
            frames.append(frame)
    v_cap.release()
    return frames, frame_width, frame_height, frame_rate

def create_video_from_frames(frames, output_path, frame_width, frame_height, frame_rate):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, frame_rate, (frame_width, frame_height))
    for frame in frames:
        out.write(frame)
    out.release()

def main():
    # Example paths
    txt_file_path = "./output/testVideo1_face_info.txt"
    video_path = "./test/testVideo1.mp4"
    output_directory = "./output/"

    # Read the txt file and extract information
    with open(txt_file_path, 'r') as file:
        lines = file.readlines()

    current_person_id = None
    persons = {}

    for line in lines:
        line = line.strip()
        if line.startswith("person_"):
            current_person_id = line[:-1]
            persons[current_person_id] = {'frames': []}
        elif line.startswith("gender:"):
            persons[current_person_id]['gender'] = line.split(': ')[1]
        elif line.startswith("age:"):
            persons[current_person_id]['age'] = line.split(': ')[1]
        elif line.startswith("frames: ["):
            frames_str = line.split(': [')[1][:-1]
            frames_list = list(map(int, frames_str.split(', ')))
            persons[current_person_id]['frames'] = frames_list

    # Extract frames from the original video and save to new video for each person
    for person_id, data in persons.items():
        if 'gender' not in data:
            print(f"Warning: Gender information not found for {person_id}. Skipping.")
            continue
        if 'age' not in data:
            print(f"Warning: Age information not found for {person_id}. Skipping.")
            continue

        frames_to_extract = data['frames']
        gender = data['gender']
        age = data['age']

        frames, frame_width, frame_height, frame_rate = extract_frames_from_video(video_path, frames_to_extract)

        output_file_name = f"{person_id}_gender_{gender}_age_{age}.mp4"
        output_path = os.path.join(output_directory, output_file_name)

        create_video_from_frames(frames, output_path, frame_width, frame_height, frame_rate)
        print(f"Created video: {output_path}")

if __name__ == "__main__":
    main()
