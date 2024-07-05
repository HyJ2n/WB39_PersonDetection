#test4에서 유사도 비교해서 저장
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import os

# Initialize MTCNN for face detection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)

# Initialize InceptionResnetV1 for face recognition
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load a reference image for comparison
reference_image_path = './test/horyun.jpg'
reference_image = cv2.imread(reference_image_path)
reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)  # Convert to RGB

# Directory to save extracted faces
output_dir = './output/similar_faces/'
os.makedirs(output_dir, exist_ok=True)

# Function to extract embeddings from an image
def extract_embeddings(image):
    boxes, _ = mtcnn.detect(image)
    if boxes is not None:
        embeddings = []
        for box in boxes:
            # Convert box from [x1, y1, x2, y2] to (left, top, width, height)
            box = [int(coord) for coord in box]
            face = image[box[1]:box[3], box[0]:box[2]]
            
            # Check if face image is empty (this check can be added)
            if face.size == 0:
                continue
            
            face = cv2.resize(face, (160, 160))
            face = torch.tensor(face).unsqueeze(0).permute(0, 3, 1, 2).float().to(device) / 255.0
            embedding = resnet(face).detach().cpu().numpy().flatten()
            embeddings.append((embedding, box))
        return embeddings
    return None

# Function to compare similarity between two sets of embeddings
def compare_similarity(embeddings1, embeddings2):
    if embeddings1 is not None and embeddings2 is not None:
        similarities = cosine_similarity([e[0] for e in embeddings1], embeddings2)
        return similarities
    return None

# Load a video file
video_path = "./test/testVideo.mp4"
v_cap = cv2.VideoCapture(video_path)

# Extract reference embeddings
reference_embeddings = extract_embeddings(reference_image)
reference_embeddings = [e[0] for e in reference_embeddings]

frame_number = 0

while True:
    # Read a frame from the video
    success, frame = v_cap.read()
    if not success:
        break
    frame_number += 1

    # Convert frame to RGB (MTCNN expects RGB image)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces and extract embeddings
    forward_embeddings = extract_embeddings(frame_rgb)
    
    # Compare similarity
    if forward_embeddings is not None:
        similarity_scores = compare_similarity(forward_embeddings, reference_embeddings)
        print(f"Similarity Scores for frame {frame_number}: {similarity_scores}")
        
        # Save images with similarity score > 0.6
        for i, (embedding, box) in enumerate(forward_embeddings):
            if similarity_scores[i][0] > 0.6:
                face_image = frame_rgb[box[1]:box[3], box[0]:box[2]]
                output_path = os.path.join(output_dir, f'frame_{frame_number}_face_{i}.jpg')
                cv2.imwrite(output_path, cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
                print(f"Saved image: {output_path}")
    
    # Draw bounding boxes on the frame and display similarity scores
    if forward_embeddings is not None:
        boxes = [e[1] for e in forward_embeddings]
        for i, box in enumerate(boxes):
            # Draw bounding box
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
            
            # Display similarity score above bounding box
            if similarity_scores is not None and i < len(similarity_scores):
                score_text = f"Similarity: {similarity_scores[i][0]:.2f}"
                cv2.putText(frame, score_text, (int(box[0]), int(box[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Display the frame with bounding boxes and similarity scores
    cv2.imshow('Frame with Bounding Boxes', frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
v_cap.release()
cv2.destroyAllWindows()
