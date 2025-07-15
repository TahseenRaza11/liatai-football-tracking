
# ⚽ Football Player Detection & Re-Identification with YOLOv11

This project showcases a complete pipeline for **football player detection**, **re-identification**, and **tracking** in match footage using a custom-trained YOLOv11 model. The system detects players, referees, and the ball, and maintains unique IDs across frames for consistent tracking.

---

## 📁 Project Structure

```

.
├── input\_video/               # Input video clips
├── model/                     # YOLOv11 trained weights (link below)
│   └── best.pt                # Custom trained weights
├── output\_videos/            # Annotated output videos
├── tracker\_stubs/            # Saved .pkl tracking metadata
├── trackers/                 # ByteTrack + Annotation logic
├── utils/                    # Helper functions
├── yolo\_inference.py         # Basic inference script
├── main.py                   # Full pipeline for tracking & annotation
└── README.md                 # This file

````

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/YourUsername/liatai-football-tracking.git
cd liatai-football-tracking
````

### 2️⃣ Create a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# OR
source venv/bin/activate  # macOS/Linux
```

### 3️⃣ Install Python Dependencies

```bash
pip install -r requirements.txt
```

If you face Torch compatibility issues:

```bash
pip install opencv-python ultralytics torch torchvision numpy pillow scikit-learn
```

---

## 🧠 Model Setup

### 🔗 Download the Model

Please download the trained model from the link below and place it inside the `model/` folder:

📥 [Google Drive – Download best.pt](https://drive.google.com/drive/folders/1WuVLSAD9GuLVVet5v49hbRaoH6-ad1wR)

Once downloaded:

```
model/
└── best.pt
```

---

## ▶️ Running the Code

### 1. YOLO Inference Only

Run basic object detection using YOLOv11:

```bash
python yolo_inference.py
```

Annotated video is saved under:
`runs/detect/predict/`

---

### 2. Full Tracking Pipeline

For player re-identification and tracking across frames:

```bash
python main.py
```

* 🟩 Output video: `output_videos/output.avi`
* 🗂️ Frame-wise tracking data: `tracker_stubs/player_detection.pkl`

---

## ✨ Features

* 🎯 Custom-trained YOLOv11 detection for:

  * Football players
  * Referees
  * Ball
* 🔢 ByteTrack-based re-identification for consistent tracking
* 📝 Annotation overlays with player IDs
* 📦 Save/load tracking metadata via `.pkl` stubs
* 🔁 Easy retraining & integration

---

## 🔧 Dependencies

* Python ≥ 3.8
* Ultralytics (YOLOv8+)
* OpenCV
* NumPy
* Torch & TorchVision
* Supervision
* scikit-learn

---

## 🧪 Future Improvements

* Integrate jersey number OCR for player identity
* Support multi-camera player tracking
* Add interactive UI to query player stats

---

## 📬 Contact

**Developer:** Tahseen Raza
📧 Email: [tahseenraza1843@gmail.com](mailto:tahseenraza1843@gmail.com)
🔗 GitHub: [@TahseenRaza11](https://github.com/TahseenRaza11)
🔗 LinkedIn: [Tahseen Raza](https://www.linkedin.com/in/tahseen-raza-11a276218/)

```
