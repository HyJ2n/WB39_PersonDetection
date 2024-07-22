import os
import pymysql
import subprocess


# 트래킹 처리 영상 DB저장 테스트

# 데이터베이스 연결 함수
def get_db_connection():
    return pymysql.connect(
        host='localhost',
        user='root',
        password='1234',
        database='wb39_project',
        cursorclass=pymysql.cursors.DictCursor
    )

# 트래킹 영상 DB에 저장 함수
def save_processed_video_info():
    try:
        # ./extracted_images 디렉토리에서 _clip이 포함된 폴더를 찾음
        extracted_dir = './extracted_images'
        clip_folders = [f for f in os.listdir(extracted_dir) if '_clip' in f]
        connection = get_db_connection()
        try:
            with connection.cursor() as cursor:
                for clip_folder in clip_folders:
                    video_name = clip_folder.replace('_clip', '')
                    clip_folder_path = os.path.join(extracted_dir, clip_folder)
                    
                    for person_id in os.listdir(clip_folder_path):
                        person_folder_path = os.path.join(clip_folder_path, person_id)
                        
                        if os.path.isdir(person_folder_path):
                            video_files = [vf for vf in os.listdir(person_folder_path) if vf.endswith('.mp4')]
                            
                            for video_file in video_files:
                                pro_video_name = f"{video_name}_{video_file}"
                                pro_video_path = os.path.abspath(os.path.join(person_folder_path, video_file))
                                
                                sql = """
                                    INSERT INTO processed_video (or_video_id, pro_video_name, pro_video_content, user_no)
                                    VALUES (%s, %s, %s, %s)
                                """
                                # 예시로 or_video_id와 user_no를 1로 설정. 실제 값을 사용해야 함.
                                cursor.execute(sql, (1, pro_video_name, pro_video_path, 1))
                connection.commit()
        except pymysql.MySQLError as e:
            print(f"MySQL error occurred: {str(e)}")
        finally:
            connection.close()
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

# 테스트를 위한 가상 데이터 및 디렉토리 생성
def create_test_environment():
    os.makedirs('./extracted_images/testVideo1_clip/person_1', exist_ok=True)
    os.makedirs('./extracted_images/testVideo1_clip/person_2', exist_ok=True)
    
    with open('./extracted_images/testVideo1_clip/person_1/person_1_output.mp4', 'w') as f:
        f.write('Dummy video content')
        
    with open('./extracted_images/testVideo1_clip/person_2/person_2_output.mp4', 'w') as f:
        f.write('Dummy video content')

if __name__ == '__main__':
    # 테스트 환경 설정
    #create_test_environment()
    
    # DB에 저장 함수 호출
    save_processed_video_info()
