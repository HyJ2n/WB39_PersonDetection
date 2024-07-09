#test 10_5_2 파일 모듈화
import cv2
import torch
import os
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1

class FaceRecognizer:
    def __init__(self, device=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.mtcnn = MTCNN(keep_all=True, post_process=False, device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.persons = []
        self.person_id = 0

    def extract_embeddings(self, image):
        boxes, probs = self.mtcnn.detect(image)
        if boxes is not None:
            embeddings = []
            for box, prob in zip(boxes, probs):
                if prob < 0.99:
                    continue
                box = [int(coord) for coord in box]
                face = image[box[1]:box[3], box[0]:box[2]]
                if face.size == 0:
                    continue
                face_tensor = torch.tensor(face).unsqueeze(0).permute(0, 3, 1, 2).float().to(self.device) / 255.0
                face_resized = torch.nn.functional.interpolate(face_tensor, size=(160, 160), mode='bilinear')
                embedding = self.resnet(face_resized).detach().cpu().numpy().flatten()
                embeddings.append((embedding, box, face, prob))
            return embeddings
        return None

    @staticmethod
    def compare_similarity(embedding1, embedding2):
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

    def recognize_faces(self, frame, frame_number, output_dir):
        original_shape = frame.shape[:2]
        frame_resized = cv2.resize(frame, (640, 480))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        forward_embeddings = self.extract_embeddings(frame_rgb)

        if forward_embeddings is not None:
            for embedding, box, face_image, prob in forward_embeddings:
                matched = False
                for person in self.persons:
                    if self.compare_similarity(embedding, person['embedding']) > 0.7:
                        self.person_id = person['id']
                        matched = True
                        break
                if not matched:
                    self.person_id = len(self.persons) + 1
                    self.persons.append({'id': self.person_id, 'embedding': embedding})

                output_path = os.path.join(output_dir, f'person_{self.person_id}_frame_{frame_number}.jpg')
                cv2.imwrite(output_path, cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
                print(f"Saved image: {output_path}, person ID: {self.person_id}, detection probability: {prob}")

            scale_x = original_shape[1] / 640
            scale_y = original_shape[0] / 480

            for embedding, box, face_image, prob in forward_embeddings:
                box = [int(coord) for coord in box]
                box[0] = int(box[0] * scale_x)
                box[1] = int(box[1] * scale_y)
                box[2] = int(box[2] * scale_x)
                box[3] = int(box[3] * scale_y)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
                id_text = f"ID: {self.person_id}"
                cv2.putText(frame, id_text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return frame

def process_video(video_path, output_dir, target_fps=10):
    v_cap = cv2.VideoCapture(video_path)
    frame_rate = int(v_cap.get(cv2.CAP_PROP_FPS))
    frame_interval = frame_rate // target_fps

    recognizer = FaceRecognizer()
    frame_number = 0

    while True:
        success, frame = v_cap.read()
        if not success:
            break
        frame_number += 1

        if frame_number % frame_interval != 0:
            continue

        frame = recognizer.recognize_faces(frame, frame_number, output_dir)
        cv2.imshow('Frame with Bounding Boxes', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    v_cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    output_dir = './output/testimage2/'
    os.makedirs(output_dir, exist_ok=True)
    video_path = "./test/testVideo.mp4"
    process_video(video_path, output_dir)
