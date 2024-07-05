#multiple faces detection and print similarity

from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
from PIL import Image, ImageDraw
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch

# Initialize MTCNN for face detection
mtcnn = MTCNN(select_largest=False, device='cuda')

# Initialize InceptionResnetV1 for face recognition
resnet = InceptionResnetV1(pretrained='vggface2').eval().to('cuda')

# Load a reference image for comparison
reference_image_path = './test/horyun.jpg'
reference_image = Image.open(reference_image_path).convert('RGB')

# Function to extract embeddings from an image
def extract_embeddings(image):
    boxes, _ = mtcnn.detect(image)
    if boxes is not None:
        embeddings = []
        for box in boxes:
            # Ensure box coordinates are integers
            box = [int(coord) for coord in box]
            
            # Crop face from image
            face = image.crop(box)
            
            # Convert PIL Image to tensor
            face = face.resize((160, 160), Image.BILINEAR)
            face_tensor = torch.tensor(np.array(face)).permute(2, 0, 1).unsqueeze(0).float().to(device='cuda') / 255.0
            embedding = resnet(face_tensor).detach().cpu().numpy().flatten()
            embeddings.append(embedding)
        
        return embeddings
    
    return None

# Function to compare similarity between two sets of embeddings
def compare_similarity(embeddings1, embeddings2):
    if embeddings1 is not None and embeddings2 is not None:
        similarities = cosine_similarity(embeddings1, embeddings2)
        return similarities
    
    return None

# Load a video file
video_path = r"C:\Users\user\Desktop\사진\KakaoTalk_20240624_153034158.mp4"
v_cap = cv2.VideoCapture(video_path)

while True:
    # Read a frame from the video
    success, frame = v_cap.read()
    if not success:
        break
    
    # Convert frame to RGB (MTCNN expects RGB image)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert frame to PIL Image
    frame_pil = Image.fromarray(frame_rgb)
    
    # Detect faces and extract embeddings
    forward_embeddings = extract_embeddings(frame_pil)
    reference_embeddings = extract_embeddings(reference_image)
    
    # Compare similarity
    if forward_embeddings is not None and reference_embeddings is not None:
        similarity_scores = compare_similarity(forward_embeddings, reference_embeddings)
        print(f"Similarity Scores: {similarity_scores}")
    
    # Draw bounding boxes on the frame
    if forward_embeddings is not None:
        draw = ImageDraw.Draw(frame_pil)
        boxes, _ = mtcnn.detect(frame_pil)
        if boxes is not None:
            for box in boxes:
                draw.rectangle(box.tolist(), outline=(255, 0, 0), width=2)
    
    # Convert PIL Image back to numpy array for displaying with cv2
    frame_with_boxes = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
    
    # Display the frame with bounding boxes
    cv2.imshow('Frame with Bounding Boxes', frame_with_boxes)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
v_cap.release()
cv2.destroyAllWindows()
