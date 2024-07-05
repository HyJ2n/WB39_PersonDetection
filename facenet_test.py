#show detect faces boundary

from facenet_pytorch import MTCNN
import cv2
from PIL import Image, ImageDraw
import numpy as np

# Create face detector
mtcnn = MTCNN(select_largest=False, device='cuda')

# Load a video file
video_path = r"C:\Users\user\Desktop\사진\KakaoTalk_20240627_103038282.mp4"
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
    
    # Detect face
    face_boxes, _ = mtcnn.detect(frame_pil)
    
    if face_boxes is not None:
        # Draw bounding boxes on the frame
        draw = ImageDraw.Draw(frame_pil)
        for box in face_boxes:
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
