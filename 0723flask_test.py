from flask import Flask, request, jsonify
import os, base64
import pymysql
import subprocess
import threading
import re

app = Flask(__name__)

# 데이터베이스 연결 함수
def get_db_connection():
    return pymysql.connect(
        host='localhost',
        user='root',
        password='1234',
        database='wb39_project',
        cursorclass=pymysql.cursors.DictCursor
    )

# 파일 저장 경로 설정
VIDEO_SAVE_PATH = 'uploaded_videos'
IMAGE_SAVE_PATH = 'uploaded_images'

# 디렉토리 생성
os.makedirs(VIDEO_SAVE_PATH, exist_ok=True)
os.makedirs(IMAGE_SAVE_PATH, exist_ok=True)

#user_id로 user_no정보 가져오기
def get_user_no(user_id):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            sql = "SELECT user_no FROM user WHERE user_id = %s"
            cursor.execute(sql, (user_id,))
            result = cursor.fetchone()
            if result:
                return result['user_no']
            else:
                print(f"No record found for user_id: {user_id}")
                return None
    except pymysql.MySQLError as e:
        print(f"MySQL error occurred: {str(e)}")
        return None
    finally:
        connection.close()

#or_video_id 통해 pro_video_id 정보 가져오기
def get_pro_video_id(or_video_id):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            sql = "SELECT pro_video_id FROM processed_video WHERE or_video_id = %s"
            cursor.execute(sql, (or_video_id,))
            result = cursor.fetchone()
            if result:
                return result['pro_video_id']
            else:
                print(f"No record found for or_video_id: {or_video_id}")
                return None
    except pymysql.MySQLError as e:
        print(f"MySQL error occurred: {str(e)}")
        return None
    finally:
        connection.close()

#현재 처리중인 비디오의 원본 ID 받아오기
def get_or_video_id(video_name):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            sql = "SELECT or_video_id FROM origin_video WHERE or_video_name = %s"
            cursor.execute(sql, (video_name,))
            result = cursor.fetchone()
            if result:
                return result['or_video_id']
            else:
                print(f"No record found for video_name: {video_name}")
                return None
    except pymysql.MySQLError as e:
        print(f"MySQL error occurred: {str(e)}")
        return None
    finally:
        connection.close()

# 메모장에서 데이터 읽기 및 파싱 함수
def parse_info_file(file_path):
    person_info = []
    with open(file_path, 'r') as file:
        content = file.read()
        persons = content.split('person_')[1:]
        for person in persons:
            person_id_match = re.search(r'(\d+):', person)
            gender_match = re.search(r'gender: (\w+)', person)
            age_match = re.search(r'age: (\w+)', person)
            if person_id_match and gender_match and age_match:
                person_id = person_id_match.group(1)
                gender = gender_match.group(1)
                age = age_match.group(1)
                person_info.append({
                    'person_id': person_id,
                    'gender': gender,
                    'age': age
                })
    return person_info

# 데이터베이스에 데이터 저장 함수
def save_to_db(person_info, pro_video_id, user_no):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            for person in person_info:
                sql = """
                    INSERT INTO person (person_id, pro_video_id, person_age, person_gender, person_color, person_clothes, person_face, person_origin_face, user_no)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                cursor.execute(sql, (
                    person['person_id'],
                    pro_video_id,
                    person['age'],
                    person['gender'],
                    '',  # person_color
                    '',  # person_clothes
                    '',  # person_face
                    '',  # person_origin_face
                    user_no
                ))
        connection.commit()
    except pymysql.MySQLError as e:
        print(f"MySQL error occurred: {str(e)}")
    finally:
        connection.close()

# 클립 처리 함수
def clip_videos(video_names, user_id, or_video_ids):
    try:
        user_no = get_user_no(user_id)
        if user_no is not None:
            for video_name, or_video_id in zip(video_names, or_video_ids):
                pro_video_id = get_pro_video_id(or_video_id)
                if pro_video_id is not None:
                    process = subprocess.Popen(
                        ["python", "videoclip_rect_flask2.py", video_name, str(user_id), str(user_no), str(pro_video_id)], 
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE
                    )
                    stdout, stderr = process.communicate()
                        
                    if process.returncode != 0:
                        print(f"Error occurred: {stderr.decode('utf-8')}")
                    else:
                        print(f"클립 추출 성공 for video {video_name}")
            
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

# 트래킹 처리 함수
def tracking_video(video_name, user_id, or_video_id):
    try:
        process = subprocess.Popen(
            ["python", "tracking_final6.py", video_name, str(user_id)], 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print(f"Error occurred: {stderr.decode('utf-8')}")
        else:
            print("트래킹 영상 추출 성공")
            user_no = get_user_no(user_id)
            if user_no is not None:
                save_processed_video_info(video_name, user_id, user_no, or_video_id)
            
                # 예시 메모장 파일 경로
                info_file_path = f'./extracted_images/{user_id}/{video_name}_face_info.txt'

                # 파싱한 person 정보
                person_info = parse_info_file(info_file_path)

                # pro_video_id 조회
                pro_video_id = get_pro_video_id(or_video_id)
                if pro_video_id is not None:
                    # DB에 저장
                    save_to_db(person_info, pro_video_id, user_no)
    
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
    
# 트래킹 영상 DB에 저장 함수
def save_processed_video_info(video_name, user_id, user_no, or_video_id):
    try:
        # ./extracted_images 디렉토리에서 _clip이 포함된 폴더를 찾음
        extracted_dir = f'./extracted_images/{user_id}'
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
                                cursor.execute(sql, (or_video_id, pro_video_name, pro_video_path, user_no))
                connection.commit()
        except pymysql.MySQLError as e:
            print(f"MySQL error occurred: {str(e)}")
        finally:
            connection.close()
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

# 얼굴 처리 함수
def process_save_face_info(video_name, user_id, or_video_id):
    try:
        # save_face_info6.py 스크립트 호출 (백그라운드 실행)
        process = subprocess.Popen(
            ["python", "save_face_info6.py", video_name, str(user_id)], 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            print(f"Error occurred: {stderr.decode('utf-8')}")
        else:
            print("정보 추출 성공")
            tracking_video(video_name, user_id, or_video_id)
            clip_video(video_name, user_id, or_video_id)

    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

# 비디오 처리 함수
def process_video(video_name, user_id):
    try:
        # Main_image2.py 스크립트 호출 (백그라운드 실행)
        process = subprocess.Popen(
            ["python", "Main_image2.py", video_name, str(user_id)], 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            print(f"Error occurred: {stderr.decode('utf-8')}")
        else:
            print("얼굴정보추출 성공")
            # 얼굴정보추출 성공 후 save_face_info6.py 실행
            video_name_with_ext = video_name + ".mp4"  # 파일 확장자 추가
            or_video_id = get_or_video_id(video_name_with_ext)
            if or_video_id is not None:
                process_save_face_info(video_name, user_id, or_video_id)

    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")


# 1.파일 업로드 엔드포인트(Post)
@app.route('/upload_file', methods=['POST'])
def upload_file():
    try:
        data = request.get_json()

        # JSON 데이터가 제대로 수신되지 않았을 경우 확인
        if not data:
            return jsonify({"status": "error", "message": "No JSON data received"}), 400
        
        # 수신된 데이터가 문자열이 아닌 JSON 객체인지 확인
        if isinstance(data, str):
            return jsonify({"status": "error", "message": "Invalid JSON data format"}), 400

        # 유저 데이터 추출
        user_data = data.get('user_data', {})
        user_id = user_data.get('user_id', '')

        # 필터링 데이터 추출
        filter_data = data.get('filter_data', {})
        age = filter_data.get('age', '')
        gender = filter_data.get('gender', '')
        color = filter_data.get('color', '')
        type = filter_data.get('type', '')

        # 필터링 데이터 출력
        print(f"Age: {age}")
        print(f"Gender: {gender}")
        print(f"Color: {color}")
        print(f"Type: {type}")

        # user_id 기반으로 디렉토리 생성
        user_video_path = os.path.join('uploaded_videos', str(user_id))
        user_image_path = os.path.join('uploaded_images', str(user_id))
        os.makedirs(user_video_path, exist_ok=True)
        os.makedirs(user_image_path, exist_ok=True)

        # 데이터베이스 연결
        connection = get_db_connection()
        with connection.cursor() as cursor:
            # 비디오 데이터 처리
            video_data = data.get('video_data', [])
            video_names = []
            for video in video_data:
                video_name = video.get('video_name', '')
                video_content_base64 = video.get('video_content', '')
                start_time = video.get('start_time', '')
                cam_name = video.get('cam_name', '')

                if video_name and video_content_base64:
                    video_content = base64.b64decode(video_content_base64)
                    video_path = os.path.join(user_video_path, video_name)
                    absolute_video_path = os.path.abspath(video_path)  # 절대 경로로 변환

                    with open(absolute_video_path, 'wb') as video_file:
                        video_file.write(video_content)

                    # 비디오 파일 이름과 절대 경로를 데이터베이스에 삽입
                    sql = """
                        INSERT INTO origin_video (or_video_name, or_video_content, start_time, cam_name)
                        VALUES (%s, %s, %s, %s)
                    """
                    cursor.execute(sql, (video_name, absolute_video_path, start_time, cam_name))
                    video_names.append(video_name)

            # 이미지 데이터 처리
            image_data = data.get('image_data', {})
            image_name = image_data.get('image_name', '')
            image_content_base64 = image_data.get('image_content', '')

            if image_name and image_content_base64:
                image_content = base64.b64decode(image_content_base64)
                image_path = os.path.join(user_image_path, image_name)
                absolute_image_path = os.path.abspath(image_path)  # 절대 경로로 변환

                with open(absolute_image_path, 'wb') as image_file:
                    image_file.write(image_content)

                # 이미지 파일 경로를 출력
                print(f"Image: {image_name}")
                print(f"Image: {absolute_image_path}")

            # 변경사항 커밋
            connection.commit()

        # 연결 종료
        connection.close()

        # 성공적으로 처리되었다는 응답 반환
        response = jsonify({"status": "success", "message": "Data received and processed successfully"})
        response.status_code = 200

        # 각 비디오 파일에 대해 비동기 처리 호출
        for video_name in video_names:
            video_base_name = os.path.splitext(video_name)[0]  # 확장자를 제거한 이름
            threading.Thread(target=process_video, args=(video_base_name, user_id)).start()

        return response

    except ValueError as e:
        print(f"A ValueError occurred: {str(e)}")
        return jsonify({"status": "error", "message": f"A ValueError occurred: {str(e)}"}), 400
    except KeyError as e:
        print(f"A KeyError occurred: {str(e)}")
        return jsonify({"status": "error", "message": f"A KeyError occurred: {str(e)}"}), 400
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return jsonify({"status": "error", "message": f"An unexpected error occurred: {str(e)}"}), 500


# 2.회원가입 엔드포인트(Post)
@app.route('/receive_data', methods=['POST'])
def receive_data():
    data = request.get_json()
    print("Received data:", data)  # 디버깅 메시지 추가
    if data and isinstance(data, list):
        connection = get_db_connection()
        cursor = connection.cursor()
        
        try:
            for user in data:
                print("Processing user:", user)  # 디버깅 메시지 추가
                # 아이디 중복 체크
                check_sql = "SELECT * FROM user WHERE user_id = %s"
                cursor.execute(check_sql, (user['ID'],))
                result = cursor.fetchone()
                
                print("Check result:", result)  # 디버깅 메시지 추가
                if result is not None and result['user_id'] == user['ID']:
                    print(f"User ID {user['ID']} already exists")  # 디버깅 메시지 추가
                    return jsonify({"error": "이미 존재하는 ID입니다"}), 409

                # User 테이블에 삽입
                user_sql = "INSERT INTO user (user_id) VALUES (%s)"
                cursor.execute(user_sql, (user['ID'],))
                print("Inserted into user table")  # 디버깅 메시지 추가
                
                # 방금 삽입한 user_no 가져오기
                user_no = cursor.lastrowid
                print("Inserted user_no:", user_no)  # 디버깅 메시지 추가

                # Password 테이블에 삽입
                password_sql = "INSERT INTO password (user_no, password) VALUES (%s, %s)"
                cursor.execute(password_sql, (user_no, user['PW']))
                print("Inserted into password table")  # 디버깅 메시지 추가

                # Profile 테이블에 삽입
                profile_sql = "INSERT INTO profile (user_no, user_name) VALUES (%s, %s)"
                cursor.execute(profile_sql, (user_no, user['Name']))
                print("Inserted into profile table")  # 디버깅 메시지 추가
            
            connection.commit()
            print("Transaction committed")  # 디버깅 메시지 추가
        except KeyError as e:
            connection.rollback()
            print(f"KeyError: {str(e)}")  # 디버깅 메시지 추가
            return jsonify({"error": f"Invalid data format: missing {str(e)}"}), 400
        except Exception as e:
            connection.rollback()
            print(f"Exception: {str(e)}")  # 디버깅 메시지 추가
            return jsonify({"error": str(e)}), 500
        finally:
            cursor.close()
            connection.close()
            print("Connection closed")  # 디버깅 메시지 추가
        
        return jsonify({"message": "Data received and stored successfully"}), 200
    else:
        print("No data received or invalid format")  # 디버깅 메시지 추가
        return jsonify({"error": "No data received or invalid format"}), 400

# 3. 로그인 엔드포인트(Post)
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    print("Login attempt:", data)  # 디버깅 메시지 추가
    if data and 'ID' in data and 'PW' in data:
        connection = get_db_connection()
        cursor = connection.cursor()
        
        try:
            # 로그인 정보 확인
            login_sql = """
                SELECT u.user_no, u.user_id, p.password
                FROM user u
                JOIN password p ON u.user_no = p.user_no
                WHERE u.user_id = %s AND p.password = %s
            """
            cursor.execute(login_sql, (data['ID'], data['PW']))
            result = cursor.fetchone()
            
            print("Login check result:", result)  # 디버깅 메시지 추가
            if result is not None:
                user_no = result['user_no']
                user_id = result['user_id']
                
                # 사용자 이름 가져오기
                profile_sql = """
                    SELECT user_name
                    FROM profile
                    WHERE user_no = %s
                """
                cursor.execute(profile_sql, (user_no,))
                profile_result = cursor.fetchone()
                user_name = profile_result['user_name'] if profile_result else "Unknown"

                # user_id를 출력
                print(f"Logged in user_id: {user_id}")

                return jsonify({
                    "message": "Login successful",
                    "user_id": user_id,
                    "user_name": user_name
                }), 200
            else:
                return jsonify({"error": "Invalid ID or password"}), 401
        except Exception as e:
            print(f"Exception: {str(e)}")  # 디버깅 메시지 추가
            return jsonify({"error": str(e)}), 500
        finally:
            cursor.close()
            connection.close()
            print("Connection closed")  # 디버깅 메시지 추가
    else:
        print("No data received or invalid format")  # 디버깅 메시지 추가
        return jsonify({"error": "No data received or invalid format"}), 400

# 4.지도 주소 엔드포인트(Post)
@app.route('/upload_maps', methods=['POST'])
def upload_map():
    try:
        data = request.get_json()
        if not data:
            print("No JSON data received")
            return jsonify({"error": "No JSON data received"}), 400

        print(f"Received data: {data}")

        connection = get_db_connection()
        print("Database connection established")
        with connection.cursor() as cursor:
            user_id = data.get('user_id')
            address = data.get('address')
            map_latitude = data.get('map_latitude')
            map_longitude = data.get('map_longitude')

            if user_id and map_latitude is not None and map_longitude is not None and address:
                # user_id를 이용하여 user_no 조회
                cursor.execute("SELECT user_no FROM user WHERE user_id = %s", (user_id))
                result = cursor.fetchone()

                if not result:
                    print(f"user_id {user_id} not found")
                    return jsonify({"error": "user_id not found"}), 404

                user_no = result['user_no']
                print(f"Found user_no: {user_no}")

                # map 테이블에 데이터 삽입
                sql = """
                    INSERT INTO map (address, map_latitude, map_longitude, user_no)
                    VALUES (%s, %s, %s, %s)
                """
                cursor.execute(sql, (address, map_latitude, map_longitude, user_no))
                print(f"Inserted: {address}, {map_latitude}, {map_longitude}, {user_no}")

                connection.commit()
                print("Transaction committed")

                cursor.execute("SELECT LAST_INSERT_ID() as map_num")
                result = cursor.fetchone()
                map_num = result['map_num']
                print(f"Retrieved map_num: {map_num}")

            else:
                print("Invalid data:", data)
                return jsonify({"error": "Invalid data format received"}), 400

        connection.close()
        print("Database connection closed")
        return jsonify({"message": "Map saved successfully", "map_num": map_num}), 200

    except pymysql.MySQLError as e:
        print(f"MySQL error occurred: {str(e)}")
        return jsonify({"error": f"MySQL error occurred: {str(e)}"}), 500
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# 5.지도 마커 위치 엔드포인트(Post)
@app.route('/upload_markers', methods=['POST'])
def upload_cameras():
    try:
        data = request.get_json()
        if not data:
            print("No JSON data received")
            return jsonify({"error": "No JSON data received"}), 400

        if not isinstance(data, list):
            print("Invalid data format received")
            return jsonify({"error": "Invalid data format received"}), 400

        print(f"Received data: {data}")

        connection = get_db_connection()
        print("Database connection established")
        with connection.cursor() as cursor:
            cursor.execute("SELECT map_num FROM map ORDER BY map_num DESC LIMIT 1")
            result = cursor.fetchone()
            if result:
                map_num = result['map_num']
                print(f"Retrieved latest map_num: {map_num}")
            else:
                print("No map records found")
                return jsonify({"error": "No map records found"}), 400

            existing_names = set()
            for camera in data:
                original_name = camera.get('name')
                cam_name = original_name
                count = 1
                while cam_name in existing_names:
                    cam_name = f"{original_name}_{count}"
                    count += 1
                existing_names.add(cam_name)

                maker_latitude = camera.get('latitude')
                maker_longitude = camera.get('longitude')

                if cam_name and maker_latitude and maker_longitude:
                    sql = """
                        INSERT INTO camera (cam_name, map_num, maker_latitude, maker_longitude)
                        VALUES (%s, %s, %s, %s)
                    """
                    cursor.execute(sql, (cam_name, map_num, maker_latitude, maker_longitude))
                    print(f"Inserted: {cam_name}, {map_num}, {maker_latitude}, {maker_longitude}")
                else:
                    print("Invalid camera data:", camera)
                    return jsonify({"error": "Invalid camera data format received"}), 400

            connection.commit()
            print("Transaction committed")

        connection.close()
        print("Database connection closed")
        return jsonify({"message": "Cameras saved successfully"}), 200

    except pymysql.MySQLError as e:
        print(f"MySQL error occurred: {str(e)}")
        return jsonify({"error": f"MySQL error occurred: {str(e)}"}), 500
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    print("Starting server")  # 서버 시작 디버깅 메시지
    app.run(debug=True)
