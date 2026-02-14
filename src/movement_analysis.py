import cv2
import os
import numpy as np
import pandas as pd
from ultralytics import YOLO
from PIL import ImageFont, ImageDraw, Image
import time

# --- [Threshold Settings for Behavioral Analysis] ---
STATIONARY_DIST = 3.0    # Threshold (pixels): Movement less than this is considered stationary
WARNING_FRAMES = 150     # Approx 5 seconds (at 30fps): Status changes to 'Stationary' (Yellow)
DEAD_FRAMES = 450        # Approx 15 seconds: Status changes to 'Dead' (Red)
# ---------------------------------------------------

def put_text_on_image(img, text, position, font_size=25, font_color=(0, 255, 0)):
    # Convert OpenCV image (NumPy array) to PIL image for text rendering
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    try:
        # Load font (Default: Malgun Gothic for Windows)
        font = ImageFont.truetype("malgun.ttf", font_size)
    except:
        # Fallback to default if font loading fails
        font = ImageFont.load_default()
    # PIL uses RGB; convert BGR font_color to RGB
    draw.text(position, text, font=font, fill=font_color[::-1])
    return np.array(img_pil)

def calculate_iou(box1, box2):
    # Calculate Intersection over Union (IoU) for object tracking
    x1, y1, x2, y2 = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
    if x2 < x1 or y2 < y1: return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    return inter / (area1 + area2 - inter)

# ==== 1. Configuration Parameters ====
# Use relative paths for GitHub portability
video_path = r".\data\video.mp4"
model_path = r".\data\best.pt"
output_video_path = "kokofarm_final_demo.mp4"
output_csv_path = "chicken_status_log.csv"

grid_size = 3
selected_cell = (2, 0)  # Row, Column of the target grid cell
conf_threshold = 0.25

if not os.path.exists(video_path): 
    exit(f"❌ Video file not found: {video_path}")

# Load YOLO model and initialize video capture
model = YOLO(model_path)
cap = cv2.VideoCapture(video_path)

# Extract video metadata
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps)

# Output VideoWriter setup
cell_w, cell_h = width // grid_size, height // grid_size
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (cell_w, cell_h))

# Tracking & Memory initialization
last_detections = []
id_counter = 0
# Memory structure: {id: {'last_center': (x,y), 'stop_count': 0, 'status': 'Active'}}
chicken_memory = {} 
activity_logs = []

print("🚀 Starting mortality analysis and logging...")

# ==== 2. Main Processing Loop ====
while True:
    ret, frame = cap.read()
    if not ret: break

    # Grid Division: Extract the specific frame area based on selected_cell
    r, c = selected_cell
    cell_frame = frame[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w].copy()

    # Object Detection
    results = model.predict(source=cell_frame, conf=conf_threshold, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    new_detections = []
    
    for i, box in enumerate(boxes):
        # 1. ID Association (Tracking Logic)
        best_iou, best_id = 0, -1
        for prev_box, prev_id in last_detections:
            iou = calculate_iou(box, prev_box)
            if iou > best_iou:
                best_iou, best_id = iou, prev_id
        
        # Match found if IoU > 0.45; otherwise, assign a new ID
        if best_iou > 0.45:
            obj_id = best_id
        else:
            obj_id = id_counter
            id_counter += 1

        # 2. Movement Analysis (Euclidean Distance)
        curr_center = ((box[0]+box[2])/2, (box[1]+box[3])/2)
        
        if obj_id not in chicken_memory:
            chicken_memory[obj_id] = {'last_center': curr_center, 'stop_count': 0, 'status': 'Active'}
        
        prev_center = chicken_memory[obj_id]['last_center']
        dist = np.sqrt((curr_center[0]-prev_center[0])**2 + (curr_center[1]-prev_center[1])**2)
        
        # 🚩 State Recovery Logic: Reset stop_count if movement is detected
        if dist < STATIONARY_DIST:
            chicken_memory[obj_id]['stop_count'] += 1
        else:
            # If the chicken moves, reset its stationary counter and status to 'Active'
            chicken_memory[obj_id]['stop_count'] = 0
            chicken_memory[obj_id]['status'] = 'Active'
        
        chicken_memory[obj_id]['last_center'] = curr_center

        # 3. Determine Status and Visualization Color
        status = 'Active'
        color = (0, 255, 0) # Green (Healthy/Moving)
        
        if chicken_memory[obj_id]['stop_count'] >= DEAD_FRAMES:
            status = 'Dead'
            color = (0, 0, 255) # Red (Potential Mortality)
        elif chicken_memory[obj_id]['stop_count'] >= WARNING_FRAMES:
            status = 'Stationary'
            color = (0, 255, 255) # Yellow (Stationary/Resting)
        
        chicken_memory[obj_id]['status'] = status

        # 4. Drawing Bounding Boxes and Status Labels
        cv2.rectangle(cell_frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
        cv2.putText(cell_frame, f"ID {obj_id} [{status}]", (int(box[0]), int(box[1])-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        new_detections.append((box, obj_id))
        
        # Collect log data for CSV export
        activity_logs.append([int(cap.get(cv2.CAP_PROP_POS_FRAMES)), obj_id, status])

    # Overlay cell info and object count on the frame
    cell_frame = put_text_on_image(cell_frame, f"Cell: {selected_cell} | Count: {len(boxes)}", (10, 30))
    
    # Write processed frame to output video
    out.write(cell_frame)
    cv2.imshow("Chicken Mortality Analysis", cell_frame)
    
    # Update detection history for next frame
    last_detections = new_detections
    if cv2.waitKey(delay) & 0xFF == ord('q'): break

# ==== 3. Resource Release & Data Export ====
cap.release()
out.release()
cv2.destroyAllWindows()

# Export activity logs to CSV using Pandas
df = pd.DataFrame(activity_logs, columns=['Frame', 'Chicken_ID', 'Status'])
df.to_csv(output_csv_path, index=False)

print(f"✅ Analysis Complete!")
print(f"- Demo video saved: {output_video_path}")
print(f"- Activity log saved: {output_csv_path}")