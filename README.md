# 🌱 Seed Germination Detection Using AI

### 🤖 Deep Learning-Based Automated Germination Analysis System

![Python](https://img.shields.io/badge/Python-3.9-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![YOLO](https://img.shields.io/badge/YOLO-Detection-green)
![DINOv2](https://img.shields.io/badge/DINOv2-Transformer-orange)
![Status](https://img.shields.io/badge/Status-Research%20Project-success)

---

## 📌 Overview

🌾 **Seed Germination Detection Using AI** is an advanced computer vision system designed to automate the process of seed germination analysis using deep learning techniques.

Traditionally, germination assessment relies on **manual observation**, which is time-consuming, labor-intensive, and prone to inconsistencies. This project addresses these limitations by introducing an intelligent pipeline that combines **object detection and vision transformer-based classification** to deliver fast, accurate, and scalable results.

🚀 The system leverages:

* 🔍 **YOLOv8** for precise seed detection
* 🧠 **DINOv2 Vision Transformer** for high-accuracy germination classification
* 🖥️ **Tkinter-based GUI** for interactive visualization and analysis

✨ By integrating detection, classification, and evaluation into a single framework, this project provides a **reliable and efficient solution for agricultural research, seed quality assessment, and automated phenotyping systems**.

---

## 📄 Research Paper

📘 **Title:** *Seed Germination Detection Using AI*

📑 **Authors:**

* Suresh Bommakola
* Sama Bhanu
* Reshma Sai
* Dr. K. Srujan Raju

📌 **DOI:**
👉 *Will be updated after publication*

---

## 🧠 Core Idea

The system combines:

* 🔍 **YOLOv8** → Detect seeds in images
* 🧠 **DINOv2 Vision Transformer** → Classify germination
* 🖥️ **Tkinter GUI** → Visualize predictions and metrics

---

## ⚙️ Project Structure

```id="k9h3sq"
SeedGermination_RT/
│
├── Dataset/
│   ├── Train/
│   │   ├── germinated/
│   │   ├── non_germinated/
│   │
│   ├── Valid/
│   ├── Test/
│
├── models/
│   ├── yolo_best.pt
│   ├── dinov2_vits14_germ.pt
│
├── sprout_env/
│
├── evaluation_ui.py
├── requirements.txt
├── run.bat
```

---

## 🔄 Workflow

```id="y2o5o4"
Input Image
    │
    ▼
YOLOv8 Detection (Seed Localization)
    │
    ▼
Crop Seeds
    │
    ▼
DINOv2 Classification
    │
    ▼
Germinated / Non-Germinated
    │
    ▼
Metrics + GUI Visualization
```

---

## 🚀 Features

✅ Automated seed detection
✅ Accurate germination classification
✅ Interactive GUI interface
✅ Confusion matrix visualization
✅ Advanced performance metrics
✅ Scalable for large datasets

---

## 📊 Evaluation Metrics

The system is evaluated using:

* 🎯 Accuracy
* 🎯 Precision
* 🎯 Recall (Sensitivity)
* 🎯 Specificity
* 🎯 F1 Score
* 🎯 MCC (Matthews Correlation Coefficient)
* 🎯 ROC-AUC
* 🎯 Confusion Matrix

---

## 🧪 Dataset

```id="1r2gqb"
Dataset/
│
├── Train/
│   ├── germinated/
│   ├── non_germinated/
│
├── Valid/
├── Test/
```

✔ Structured dataset
✔ Binary classification
✔ Ready for training and evaluation

---

## ▶️ Usage

```bash id="mq9z2l"
python evaluation_ui.py
```

---

## 📊 Sample Performance

| Metric    | Value |
| --------- | ----- |
| Accuracy  | ~95%  |
| Precision | ~94%  |
| Recall    | ~93%  |
| F1 Score  | ~94%  |

---

## 🧠 Models Used

### 🔍 YOLOv8

* Real-time object detection
* High-speed and accurate

### 🧠 DINOv2

* Vision Transformer
* Strong feature extraction
* High classification performance

---

## 📌 Applications

🌾 Agricultural research
🌱 Seed quality evaluation
🏭 Seed industry automation
📊 High-throughput phenotyping

---

## 🚧 Future Scope

* 🌐 Web-based system
* 📱 Mobile application
* 🎥 Real-time monitoring
* 🌍 Multi-seed species support

---

## 👨‍💻 Authors

* **Suresh Bommakola**
* **Sama Bhanu**
* **Reshma Sai**
* **Dr. K. Srujan Raju** *(Supervisor)*

---

## ⭐ Support

If you like this project:

⭐ Star this repository
🔁 Fork and contribute
📢 Share with others

---

💡 *“Empowering agriculture with intelligent AI-driven solutions.”* 🌱
