import cv2
from ultralytics import YOLO

# 1. Load the AI Model (Downloads a tiny 3MB file the very first time)
print("Loading SmartWaste AI...")
model = YOLO('yolov8n.pt') 

# 2. Open the Webcam (0 is the default laptop camera)
cap = cv2.VideoCapture(0)

print("Camera is starting... Show a plastic bottle to the camera!")

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to access camera.")
        break

    # 3. AI Inference: Detect objects in the frame instantly
    results = model(frame, stream=True)

    # 4. Draw boxes and labels
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get Box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Get Class name (What object is it?)
            class_name = model.names[int(box.cls[0])]
            
            # Draw the bounding box (Green box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Put the label text above the box
            cv2.putText(frame, f"Detected: {class_name}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 5. Show the video window
    cv2.imshow("SmartWaste - Live Demo", frame)

    # Press 'q' on your keyboard to close the demo
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up when done
cap.release()
cv2.destroyAllWindows()
