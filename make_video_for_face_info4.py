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

def process_video(video_path, txt_file_path, output_base_directory):
    # Extract the base name of the video file (without extension)
    video_base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_directory = os.path.join(output_base_directory, f"{video_base_name}_clip")

    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

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

        # Create person-specific directory within the output directory
        person_output_directory = os.path.join(output_directory, person_id)
        os.makedirs(person_output_directory, exist_ok=True)

        output_file_name = f"{person_id}_gender_{gender}_age_{age}.mp4"
        output_path = os.path.join(person_output_directory, output_file_name)

        create_video_from_frames(frames, output_path, frame_width, frame_height, frame_rate)
        print(f"Created video: {output_path}")

def main():
    output_base_directory = "./output"
    txt_file_paths = []
    video_paths = []

    # Find all _face_info.txt files
    for root, _, files in os.walk(output_base_directory):
        for file in files:
            if file.endswith('_face_info.txt'):
                txt_file_path = os.path.join(root, file)
                txt_file_paths.append(txt_file_path)
                
                # Derive corresponding video file path
                video_base_name = file.replace('_face_info.txt', '')
                video_file_path = os.path.join('./test', f"{video_base_name}.mp4")
                
                if os.path.exists(video_file_path):
                    video_paths.append(video_file_path)
                else:
                    print(f"Warning: Video file {video_file_path} not found for {txt_file_path}")

    # Process each video and its corresponding txt file
    for video_path, txt_file_path in zip(video_paths, txt_file_paths):
        process_video(video_path, txt_file_path, output_base_directory)

if __name__ == "__main__":
    main()
