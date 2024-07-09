#test 10_5 파일 메서드화
import cv2
import torch
import os
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1

# Initialize MTCNN for face detection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, post_process=False, device=device)

# Initialize InceptionResnetV1 for face recognition
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def create_output_dir(output_dir='./output/testimage2/'):
    os.makedirs(output_dir, exist_ok=True)

def load_reference_image(reference_image_path):
    reference_image = cv2.imread(reference_image_path)
    reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    return reference_image

def extract_embeddings(image):
    boxes, probs = mtcnn.detect(image)
    if boxes is not None:
        embeddings = []
        for box, prob in zip(boxes, probs):
            if prob < 0.99:  # Only consider faces with a high detection probability
                continue
            box = [int(coord) for coord in box]
            face = image[box[1]:box[3], box[0]:box[2]]
            
            if face.size == 0:
                continue
            
            face_tensor = torch.tensor(face).unsqueeze(0).permute(0, 3, 1, 2).float().to(device) / 255.0
            face_resized = torch.nn.functional.interpolate(face_tensor, size=(160, 160), mode='bilinear')
            embedding = resnet(face_resized).detach().cpu().numpy().flatten()
            embeddings.append((embedding, box, face, prob))
        return embeddings
    return None

def compare_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

def process_video(video_path, output_dir, reference_image, target_fps=10):
    v_cap = cv2.VideoCapture(video_path)
    frame_rate = int(v_cap.get(cv2.CAP_PROP_FPS))
    frame_interval = frame_rate // target_fps

    reference_embeddings = extract_embeddings(reference_image)
    reference_embeddings = [e[0] for e in reference_embeddings]

    frame_number = 0
    person_id = 0
    persons = []

    while True:
        success, frame = v_cap.read()
        if not success:
            break
        frame_number += 1

        if frame_number % frame_interval != 0:
            continue

        original_shape = frame.shape[:2]
        frame_resized = cv2.resize(frame, (1024, 768))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        forward_embeddings = extract_embeddings(frame_rgb)

        if forward_embeddings is not None:
            for embedding, box, face_image, prob in forward_embeddings:
                matched = False
                for person in persons:
                    if compare_similarity(embedding, person['embedding']) > 0.7:
                        person_id = person['id']
                        matched = True
                        break
                if not matched:
                    person_id = len(persons) + 1
                    persons.append({'id': person_id, 'embedding': embedding})
                
                output_path = os.path.join(output_dir, f'person_{person_id}_frame_{frame_number}.jpg')
                cv2.imwrite(output_path, cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
                print(f"Saved image: {output_path}, person ID: {person_id}, detection probability: {prob}")

            scale_x = original_shape[1] / 1024
            scale_y = original_shape[0] / 768
            
            for embedding, box, face_image, prob in forward_embeddings:
                box = [int(coord) for coord in box]
                box[0] = int(box[0] * scale_x)
                box[1] = int(box[1] * scale_y)
                box[2] = int(box[2] * scale_x)
                box[3] = int(box[3] * scale_y)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
                id_text = f"ID: {person_id}"
                cv2.putText(frame, id_text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        cv2.imshow('Frame with Bounding Boxes', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    v_cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    output_dir = './output/testimage2/'
    create_output_dir(output_dir)
    
    reference_image_path = './test/horyun.jpg'
    reference_image = load_reference_image(reference_image_path)
    
    video_path = "./test/testVideo.mp4"
    process_video(video_path, output_dir, reference_image)
