import os
import re
import pymysql


#트래킹 영상 DB에 저장후 person정보 DB에 저장 테스트

# 데이터베이스 연결 함수
def get_db_connection():
    return pymysql.connect(
        host='localhost',
        user='root',
        password='1234',
        database='wb39_project',
        cursorclass=pymysql.cursors.DictCursor
    )

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

# 테스트를 위한 가상 데이터 및 메모장 파일 생성
def create_test_file():
    os.makedirs('./extracted_images', exist_ok=True)
    test_file_content = """
    person_1:
      gender: Male
      age: Youth
      frames: [2, 4, 6, 8, 10]

    person_2:
      gender: Female
      age: Adult
      frames: [12, 14, 16, 18, 20]
    """
    with open('./extracted_images/testVideo1_face_info.txt', 'w') as f:
        f.write(test_file_content)

if __name__ == '__main__':
    # 테스트 파일 생성
    #create_test_file()
    
    # 예시 메모장 파일 경로
    info_file_path = './extracted_images/testVideo1_face_info.txt'

    # 파싱한 person 정보
    person_info = parse_info_file(info_file_path)

    # 예시 pro_video_id와 user_no
    pro_video_id = 1  # 예시 값, 실제 값으로 대체 필요
    user_no = 1       # 예시 값, 실제 값으로 대체 필요

    # DB에 저장
    save_to_db(person_info, pro_video_id, user_no)
