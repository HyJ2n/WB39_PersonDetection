import cv2
import os
import pymysql
from datetime import datetime, timedelta
import numpy as np
import sys

def get_db_connection():
    return pymysql.connect(
        host='localhost',
        user='root',
        password='1234',
        database='wb39_project',
        cursorclass=pymysql.cursors.DictCursor
    )

def get_video_start_time(index):
    start_times = [
        datetime(2024, 7, 4, 10, 1, 0),
        datetime(2024, 7, 4, 10, 0, 0),
        datetime(2024, 7, 4, 10, 0, 30)
    ]
    return start_times[index]

# 클립 저장 경로 설정
def process_clips(video_name, user_id, user_no):
    output_clips_dir = f'./extracted_images/{user_id}/output_clips'
    os.makedirs(output_clips_dir, exist_ok=True)

    # 클립 시간 정보를 저장할 리스트
    clip_times = []

    # _clip 디렉토리에서 video_path와 시작 시간 설정
    extracted_dir = f'./extracted_images/{user_id}'
    clip_folders = [f for f in os.listdir(extracted_dir) if '_clip' in f]

    # 클립 정보를 저장할 리스트
    video_path = []

    for clip_folder_index, clip_folder in enumerate(clip_folders):
        video_name = clip_folder.replace('_clip', '')
        clip_folder_path = os.path.join(extracted_dir, clip_folder)

        for person_id in os.listdir(clip_folder_path):
            person_folder_path = os.path.join(clip_folder_path, person_id)

            if os.path.isdir(person_folder_path):
                video_files = [vf for vf in os.listdir(person_folder_path) if vf.endswith('.mp4')]

                for video_file in video_files:
                    video_file_path = os.path.join(person_folder_path, video_file)
                    video_start_time = get_video_start_time(clip_folder_index)  # 설정된 시작 시간
                    person_id_number = person_id.replace('person_', '')  # 접두어 제거
                    video_path_entry = {"file": video_file_path, "start_time": video_start_time, "person_id": person_id_number}
                    video_path.append(video_path_entry)

    # 입력 비디오 파일 목록과 시작 시간 설정
    for vp in video_path:
        # 비디오 파일이 존재하는지 확인
        if not os.path.exists(vp["file"]):
            print(f"Error: Video file not found at {vp['file']}")
            continue

        # 비디오 캡처 객체 생성
        cap = cv2.VideoCapture(vp["file"])
        if not cap.isOpened():
            print(f"Error: Could not open video {vp['file']}.")
            continue

        # 비디오 프레임 속성 가져오기
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = cap.get(cv2.CAP_PROP_FPS)

        clip_start_frame = None
        clip_index = 0
        combined_writer = None
        file_label = os.path.basename(vp["file"]).split(".")[0]

        # 비디오 처리
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 바운딩 박스 색상 탐지 (파란색)
            lower_blue = np.array([0, 255, 0], dtype=np.uint8)
            upper_blue = np.array([0, 255, 0], dtype=np.uint8)
            mask = cv2.inRange(frame, lower_blue, upper_blue)
            blue_pixel_count = cv2.countNonZero(mask)

            if blue_pixel_count > 0:
                if combined_writer is None:
                    clip_filename = f'{file_label}_{clip_index}.mp4'
                    combined_clip_path = os.path.join(output_clips_dir, clip_filename)
                    combined_writer = cv2.VideoWriter(
                        combined_clip_path,
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        frame_rate,
                        (frame_width, frame_height)
                    )

                if clip_start_frame is None:
                    clip_start_frame = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
                    clip_start_time = vp["start_time"] + timedelta(seconds=(clip_start_frame / frame_rate))
                    clip_times.append({
                        "clip_file": combined_clip_path,
                        "start_time": clip_start_time,
                        "file_label": f"{file_label}_{clip_index}",  # Save label for sorting
                        "video_label": file_label,  # Save video label for path
                        "person_id": vp["person_id"]
                    })

                combined_writer.write(frame)
            else:
                if combined_writer is not None:
                    combined_writer.release()
                    combined_writer = None
                    clip_start_frame = None
                    clip_index += 1

        if combined_writer is not None:
            combined_writer.release()

        cap.release()
        cv2.destroyAllWindows()

    # 클립 시간 정보를 실제 시간 기준으로 정렬
    clip_times.sort(key=lambda x: x["start_time"])

    # 비디오 파일 경로 순서를 생성
    video_order_path = []
    for clip in clip_times:
        if not video_order_path or video_order_path[-1] != clip["video_label"]:
            video_order_path.append(clip["video_label"])

    # 클립 시간 정보를 텍스트 파일로 저장
    with open(os.path.join(output_clips_dir, 'clips_times.txt'), 'w') as f:
        for clip_info in clip_times:
            f.write(f"클립 파일: {clip_info['clip_file']}, 시작 시간: {clip_info['start_time']}, person_id: {clip_info['person_id']}\n")  # person_id 추가
        
        # 비디오 파일 순서 추가
        f.write("\n비디오 파일 순서:\n")
        video_order_text = ",".join([clip_info['file_label'] for clip_info in clip_times])
        f.write(video_order_text)
        
        # 요약된 이동 경로 순서 추가
        f.write("\n\n이동 경로:\n")
        f.write(",".join(video_order_path))

    # 정렬된 클립 시간 정보 출력
    print("클립 시간 정보 (정렬된 순서):")
    for clip_info in clip_times:
        print(f"클립 파일: {clip_info['clip_file']}, 시작 시간: {clip_info['start_time']}, person_id: {clip_info['person_id']}")  # person_id 추가

    print("클립의 시간 정보가 clips_times.txt 파일로 저장되었습니다.")

    # 클립 정보를 DB에 저장
    save_clips_to_db(clip_times, user_no)

def save_clips_to_db(clip_times, user_no):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            for clip_info in clip_times:
                sql = """
                    INSERT INTO clip (person_id, clip_video, clip_location, clip_time, user_no)
                    VALUES (%s, %s, %s, %s, %s)
                """
                cursor.execute(sql, (
                    clip_info['person_id'],
                    clip_info['clip_file'],
                    'Location Placeholder',  # 임시 위치 정보
                    clip_info['start_time'],
                    user_no
                ))
            connection.commit()
    except pymysql.MySQLError as e:
        print(f"MySQL error occurred: {str(e)}")
    finally:
        connection.close()
        print("클립 정보가 데이터베이스에 저장되었습니다.")

# 메인 함수
if __name__ == "__main__":
    try:
        video_name = sys.argv[1]
        user_id = sys.argv[2]
        user_no = int(sys.argv[3])

        process_clips(video_name, user_id, user_no)
    except Exception as e:
        print(f"An error occurred: {str(e)}", file=sys.stderr)
        sys.exit(1)
