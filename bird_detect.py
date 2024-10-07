import cv2
import datetime
import os
import torch
from ultralytics import YOLO

def square_bbox(x1, y1, x2, y2, img_width, img_height):
    width = x2 - x1
    height = y2 - y1
    size = max(width, height)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    new_x1 = max(0, int(center_x - size / 2))
    new_y1 = max(0, int(center_y - size / 2))
    new_x2 = min(img_width, int(center_x + size / 2))
    new_y2 = min(img_height, int(center_y + size / 2))
    
    return new_x1, new_y1, new_x2, new_y2

def detect_birds_in_image(image, model_path="model/yolov8n.pt", conf_threshold=0.4, save_dir="bird_crops"):
    if image is None:
        print("Error: Input image is None. Please check the image path.")
        return []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = YOLO(model_path, "v8")
    model.to(device)

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    print(f"Save directory: {os.path.abspath(save_dir)}")

    # Get original image dimensions
    img_height, img_width = image.shape[:2]
    
    # Perform detection
    detect_params = model.predict(source=[image], conf=conf_threshold, save=False)
    DP = detect_params[0].cpu().numpy()

    detected_birds = []
    highest_conf_bird = None
    highest_conf = 0

    if len(DP) == 0:
        print("No objects detected in the image.")
        return []

    for i in range(len(detect_params[0])):
        boxes = detect_params[0].boxes
        box = boxes[i]
        conf = box.conf.cpu().numpy()[0]
        bb = box.xyxy.cpu().numpy()[0]
        c = box.cls
        class_name = model.names[int(c)]

        print(f"Detected object: {class_name}, confidence: {conf:.2%}")

        if 'bird' in class_name.lower():
            # Convert bounding box to square
            x1, y1, x2, y2 = square_bbox(bb[0], bb[1], bb[2], bb[3], img_width, img_height)

            # Update highest confidence bird
            if conf > highest_conf:
                highest_conf = conf
                highest_conf_bird = {
                    'class': class_name,
                    'confidence': conf,
                    'bbox': (x1, y1, x2, y2),
                    'crop': image[y1:y2, x1:x2]
                }
                
    if highest_conf_bird:
    #     # Save only the highest confidence bird image
    #     # current_time = datetime.datetime.now()
    #     # filename = os.path.join(save_dir, f"bird_{highest_conf_bird['class']}_{current_time.strftime('%Y-%m-%d_%H-%M-%S-%f')}.jpg")
    #     # success = cv2.imwrite(filename, highest_conf_bird['crop'])
        
    #     # if success:
    #     #     print(f"Successfully saved highest confidence bird image to: {os.path.abspath(filename)}")
    #     #     highest_conf_bird['filename'] = filename
    #     #     detected_birds.append(highest_conf_bird)
    #     # else:
    #     #     print(f"Failed to save highest confidence bird image to: {os.path.abspath(filename)}")
        return highest_conf_bird['crop']
    else:
        return image
        # print("No birds detected in the image.")



# # Example usage:
# if __name__ == "__main__":
#     # Load an image
#     image_path = "Acadian_Flycatcher_0014_2619413295.jpg"
#     input_image = cv2.imread(image_path)
    
#     if input_image is None:
#         print(f"Error: Could not read image from {image_path}")
#     else:
#         print(f"Successfully loaded image from {image_path}")
        
#         # Detect birds in the image
#         detections = detect_birds_in_image(input_image)
        
#         # Print detections
#         for bird in detections:
#             print(f"Detected {bird['class']} with confidence {bird['confidence']:.2%} at {bird['bbox']}")
#             print(f"Saved cropped image to: {bird['filename']}")