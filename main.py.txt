import customtkinter as ctk
import cv2
from PIL import Image
from datetime import datetime
from ultralytics import YOLO
import os
import csv
import time

# ==========================================
# 1. SMARTWASTE MAPPING & CONFIGURATION
# ==========================================
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")

COLORS = {
    "PLASTIC": (0, 0, 255), # Red 
    "ORGANIC": (0, 255, 0), # Green 
    "PAPER": (255, 0, 0), # Blue 
    "METAL": (0, 255, 255), # Yellow 
    "HAZARDOUS": (0, 0, 255) # Red Alert 
}

WASTE_MAP = {
    'bottle': 'PLASTIC', 'cup': 'PLASTIC',
    'apple': 'ORGANIC', 'banana': 'ORGANIC', 'orange': 'ORGANIC',
    'book': 'PAPER',
    'scissors': 'METAL',
    'cell phone': 'HAZARDOUS', 'laptop': 'HAZARDOUS'
}

class SmartWasteDashboard(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("SmartWaste: AI Waste Detection System")
        self.geometry("1000x650")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # --- Sidebar ---
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="SmartWaste", font=ctk.CTkFont(size=24, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Notice the button name change here!
        self.start_btn = ctk.CTkButton(self.sidebar_frame, text="Start AI System", command=self.start_camera)
        self.start_btn.grid(row=1, column=0, padx=20, pady=10)

        self.stop_btn = ctk.CTkButton(self.sidebar_frame, text="Stop System", command=self.stop_camera, fg_color="red", hover_color="darkred")
        self.stop_btn.grid(row=2, column=0, padx=20, pady=10)
        
        # --- Main Area ---
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.main_frame.grid_rowconfigure(0, weight=4) 
        self.main_frame.grid_rowconfigure(1, weight=1) 
        self.main_frame.grid_columnconfigure(0, weight=1)

        self.video_label = ctk.CTkLabel(self.main_frame, text="SYSTEM OFFLINE", fg_color="black", font=ctk.CTkFont(size=24, weight="bold"))
        self.video_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.log_textbox = ctk.CTkTextbox(self.main_frame, font=ctk.CTkFont(size=14))
        self.log_textbox.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        # --- AI & Hardware Setup ---
        self.cap = None
        self.is_running = False
        self.log_textbox.insert("end", "[INFO] Loading YOLOv8 AI Model. Please wait...\n")
        self.model = YOLO('yolov8n.pt') 
        self.log_textbox.insert("end", "[INFO] AI Model Loaded Successfully.\n")
        
        # CSV Logging Setup
        self.log_file = "logs/waste_event_log.csv"
        self.init_csv()
        self.last_log_time = time.time()

    def init_csv(self):
        """Creates the CSV file if it doesn't exist."""
        if not os.path.exists("logs"):
            os.makedirs("logs")
        if not os.path.exists(self.log_file):
            with open(self.log_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Date", "Time", "Waste Category", "Confidence", "Action Required"])

    def log_event(self, category, confidence):
        """Logs to UI and saves to CSV (Cooldown of 3 seconds to prevent spam)."""
        current_time = time.time()
        if current_time - self.last_log_time > 3.0: 
            date_str = datetime.now().strftime("%Y-%m-%d")
            time_str = datetime.now().strftime("%H:%M:%S")
            action = "Special Disposal" if category == "HAZARDOUS" else "Recycle/Compost"
            
            # 1. Update UI
            log_msg = f"[{time_str}] DETECTED: {category} | Conf: {confidence}% | Alert: {action}\n"
            self.log_textbox.insert("end", log_msg)
            self.log_textbox.see("end")
            
            # 2. Save to Database/CSV
            with open(self.log_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([date_str, time_str, category, f"{confidence}%", action])
                
            self.last_log_time = current_time

    def start_camera(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            self.is_running = True
            self.log_textbox.insert("end", "[SYSTEM] Camera started. Monitoring for waste...\n")
            self.update_frame()

    def stop_camera(self):
        if self.is_running:
            self.is_running = False
            if self.cap:
                self.cap.release()
            self.video_label.configure(image="", text="SYSTEM OFFLINE")
            self.log_textbox.insert("end", "[SYSTEM] Camera disconnected.\n")

    def update_frame(self):
        if self.is_running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                # Run AI Inference
                results = self.model(frame, stream=True, verbose=False)
                
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        raw_name = self.model.names[int(box.cls[0])]
                        confidence = int(box.conf[0] * 100)
                        
                        if raw_name in WASTE_MAP:
                            category = WASTE_MAP[raw_name]
                            color = COLORS[category]
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            
                            # Draw Bounding Box & Label
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                            label = f"{category} {confidence}%"
                            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            cv2.rectangle(frame, (x1, y1 - 25), (x1 + w, y1), color, -1)
                            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            
                            # Trigger logging & alerts
                            self.log_event(category, confidence)
                            
                            # Visual Hazard Alert Overlay
                            if category == "HAZARDOUS":
                                cv2.putText(frame, "HAZARD DETECTED!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

                # Convert frame for UI display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(750, 480))
                self.video_label.configure(image=ctk_img, text="")
            
            self.after(15, self.update_frame)

if __name__ == "__main__":
    app = SmartWasteDashboard()
    app.mainloop()
