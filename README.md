# Facial Emotion Detection using YOLOv8 🎭

A deep learning project for real-time **facial emotion detection**, fine-tuned on the FER-2013 dataset using **YOLOv8**. This system detects and classifies 7 emotions: `Angry`, `Disgust`, `Fear`, `Happy`, `Natural`, `Sad`, and `Surprised`.

---

## 🔍 Overview

This project leverages the power of object detection using YOLOv8 to recognize facial expressions in real-time. It includes:

- Preprocessing and augmentation of 30,000+ facial images.
- Fine-tuning YOLOv8 on a curated Roboflow dataset.
- Real-time inference using webcam input.
- A student-teacher knowledge distillation pipeline in progress.

---

## 📊 Results

- **mAP@0.5**: `~94%`
- **mAP@0.5:0.95**: `~70.5%`
- Achieved high precision and recall across all 7 emotion classes.

| Emotion    | Precision | Recall | mAP@0.5 |
|------------|-----------|--------|---------|
| Angry      | 88.4%     | 98.7%  | 96.7%   |
| Disgust    | 100%      | 68.4%  | 93.4%   |
| Fear       | 94.3%     | 64.0%  | 87.6%   |
| Happy      | 84.6%     | 100%   | 93.9%   |
| Natural    | 95.3%     | 91.0%  | 97.9%   |
| Sad        | 89.9%     | 85.3%  | 94.8%   |
| Surprised  | 100%      | 85.8%  | 93.8%   |

---

## 🧠 Model Architecture

- **Model**: YOLOv8n
- **Framework**: PyTorch (Ultralytics)
- **Training**: 50 epochs on Google Colab (Tesla T4 GPU)
- **Input Size**: 640×640
- **Loss Function**: YOLO loss (objectness + classification + localization)

---

## 📁 Directory Structure

```bash
facial-emotion-detection/
├── data-primer-2-9/                # Downloaded dataset folder (from Roboflow)
│   ├── train/
│   ├── valid/
│   ├── test/
│   └── data.yaml                   # Contains path and class info
│
├── runs/                           # YOLOv8 output directory
│   └── detect/
│       └── train/                  # First training run
│       └── train2/                 # Second training run (latest results)
│           ├── weights/           # Contains best.pt and last.pt
│           └── args.yaml          # Training configuration
│
├── export/                         # Folder for saved/exported models
│   └── sign_model/ (optional)
│       ├── config.json
│       ├── pytorch_model.bin
│       └── tokenizer_config.json
│
├── yolov8n.pt                      # YOLOv8n pre-trained weights
├── inference.py                    # Script for webcam-based real-time inference
├── train.py                        # Script to trigger model training
├── requirements.txt                # Required Python packages
└── README.md                       # Project overview and instructions

```


---

## ⚙️ Installation & Setup

```bash
git clone https://github.com/your-username/facial-emotion-detection.git
cd facial-emotion-detection
pip install -r requirements.txt

```

🚀 Training
```bash
!yolo task=detect mode=train model="yolov8n.pt" data=data/data.yaml epochs=50 imgsz=640
```

📦 Inference
```bash
!yolo task=detect mode=predict model=runs/detect/train/weights/best.pt source=0
```

📌 Tech Stack
Python, YOLOv8, PyTorch, OpenCV, Google Colab, Roboflow, FER-2013 Dataset

🏁 Future Work
- Implement student-teacher distillation for lightweight deployment.
- Integrate GUI with real-time webcam support.
- Optimize for edge devices (Jetson Nano, Raspberry Pi).

🤝 Acknowledgments
- Ultralytics for YOLOv8
- Roboflow for dataset management
- FER-2013 Dataset

📬 Contact
For questions or collaboration, reach out via LinkedIn or email.
