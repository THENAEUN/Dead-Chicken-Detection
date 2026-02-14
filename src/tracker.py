# Import OpenCV for image and video processing
import cv2
# Import os for file path and directory management
import os
# Import NumPy for numerical operations and array processing
import numpy as np
# Import YOLO from ultralytics for object detection
from ultralytics import YOLO
# Import PIL modules for handling Korean text rendering (compatible with OpenCV)
from PIL import ImageFont, ImageDraw, Image

# Function to display text on images (supporting non-ASCII characters)
def put_text_on_image(img, text, position, font_size=30, font_color=(0, 255, 0)):
    # Convert OpenCV BGR image to PIL format
    img_pil = Image.fromarray(img)
    # Initialize drawing object
    draw = ImageDraw.Draw(img_pil)
    try:
        # Load font (Default: Malgun Gothic for Windows, adjust path for Linux/Mac)
        font = ImageFont.truetype("malgun.ttf", font_size)
    except:
        # Fallback to default font if the specified font is not found
        font = ImageFont.load_default()
    
    # PIL uses RGB; convert BGR font_color to RGB
    draw.text(position, text, font=font, fill=font_color[::-1])
    # Convert back to NumPy array for OpenCV compatibility
    return np.array(img_pil)

# Function to calculate Intersection over Union (IoU) between two bounding boxes
def calculate_iou(box1, box2):
    # Calculate intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Return 0.0 if there is no overlap
    if x2 < x1 or y2 < y1:
        return 0.0
    
    # Calculate intersection and union areas
    inter_area = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter_area
    
    return inter_area / union

# === 1. Configuration ===
# Path to the input video file (relative path for GitHub portability)
video_path = r".\kokofarm\code\video.mp4"
# Path to the YOLO weight file
model_path = r".\kokofarm\code\best.pt"
# Name of the output video file
output_path = "kokofarm_tracking_demo.mp4"

# Grid settings (split frame into NxN cells)
grid_size = 3
selected_cell = (2, 0)  # (row, col)
scale_factor = 1.0
conf_threshold = 0.5    # Confidence threshold for detection

# === 2. Initialization & Video Writer Setup ===
# Check if video file exists
if not os.path.exists(video_path):
    print(f"❌ File not found: {video_path}")
    exit()

# Load YOLO model
model = YOLO(model_path)
# Initialize video capture
cap = cv2.VideoCapture(video_path)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps)

# Calculate dimensions for the selected cell
cell_h = height // grid_size
cell_w = width // grid_size

# Configure VideoWriter (Codec: mp4v)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (cell_w, cell_h))

# List to store detections from the previous frame for ID recovery
last_detections = []

# === 3. Main Processing Loop ===
print(f"🚀 Starting analysis and recording: {output_path}")

while True:
    # Read frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Grid division: Extract selected cell
    row, col = selected_cell
    x, y = col * cell_w, row * cell_h
    cell_frame = frame[y:y+cell_h, x:x+cell_w].copy()

    # Perform YOLO tracking (imgsz=640 for consistency and accuracy)
    results = model.track(cell_frame, conf=conf_threshold, persist=True, verbose=False, imgsz=640)[0]
    
    # Extract bounding boxes and object IDs
    boxes = results.boxes.xyxy.cpu().tolist()
    ids = results.boxes.id.int().cpu().tolist() if results.boxes.id is not None else [-1] * len(boxes)

    new_detections = []
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        obj_id = ids[i]

        # ID Recovery logic using IoU if tracker fails to assign an ID
        if obj_id == -1:
            for last_box, last_id in last_detections:
                iou = calculate_iou(box, last_box)
                if iou > 0.5:
                    obj_id = last_id
                    break
        
        new_detections.append(((x1, y1, x2, y2), obj_id))

    # === Visualization ===
    plotted = cell_frame.copy()
    
    for (x1, y1, x2, y2), obj_id in new_detections:
        # Draw bounding boxes and ID labels
        cv2.rectangle(plotted, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(plotted, f"ID {obj_id}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    # Add overlay text for cell info and object count
    chicken_count = len(new_detections)
    plotted = put_text_on_image(plotted, f"Cell {selected_cell} - Detections: {chicken_count}", (10, 30), 25)

    # Save the processed frame to output file
    out.write(plotted)

    # Display the output window
    cv2.imshow("Chicken Tracking Demo", plotted)

    # Update previous detection records
    last_detections = [((x1, y1, x2, y2), obj_id) for (x1, y1, x2, y2), obj_id in new_detections]

    # Break loop if 'q' is pressed
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

# === 5. Resource Release ===
cap.release()
out.release()  # Ensure VideoWriter is properly closed to save the file
cv2.destroyAllWindows()

print(f"✅ Processing complete. Output saved to: {os.path.abspath(output_path)}")
