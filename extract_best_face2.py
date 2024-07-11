import os
import shutil
import cv2

def compute_image_quality(image):
    # Compute quality score of the image using variance of Laplacian
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score

def organize_faces_by_person(input_folder, output_folder):
    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    person_images = {}

    # Iterate through files in the input folders
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            filepath = os.path.join(input_folder, filename)

            # Extract person_id from filename assuming format is "person_{person_id}_..."
            person_id = filename.split('_')[1]  # This splits the filename by '_' and gets the second part

            # Read image
            image = cv2.imread(filepath)
            quality_score = compute_image_quality(image)

            if person_id not in person_images:
                person_images[person_id] = (filepath, quality_score)
            else:
                # Compare and keep the image with the highest quality score
                if quality_score > person_images[person_id][1]:
                    person_images[person_id] = (filepath, quality_score)

    # Save the highest quality images to the respective person_id folders
    for person_id, (best_image_path, _) in person_images.items():
        person_folder = os.path.join(output_folder, f"person_{person_id}")
        os.makedirs(person_folder, exist_ok=True)
        output_filepath = os.path.join(person_folder, os.path.basename(best_image_path))
        shutil.copy(best_image_path, output_filepath)
        print(f"Copied best quality face image: {output_filepath}")

if __name__ == "__main__":
    # List of (input_folder, output_folder) pairs
    folder_pairs = [
        ("./output/testVideo1_face", "./output/testVideo1_clip"),
        ("./output/testVideo2_face", "./output/testVideo2_clip")
    ]

    # Process each input-output folder pair
    for input_folder, output_folder in folder_pairs:
        organize_faces_by_person(input_folder, output_folder)
