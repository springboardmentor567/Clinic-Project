# 🩺 CliniScan: Lung-Abnormality Detection on Chest X-rays using AI

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?logo=keras)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Deployed-success)

---

## 📘 Overview

**CliniScan** is an AI-powered web application that detects **lung abnormalities** such as **Pneumonia** from **Chest X-ray images**.
Built using **Deep Learning** and **Transfer Learning (VGG19)**, the model analyzes medical X-rays and classifies them as **Normal** or **Pneumonia**, assisting healthcare professionals in early diagnosis.

---

## 🚀 Features

* 🧠 **AI-based Diagnosis** — Uses VGG19 (a pre-trained CNN) for accurate lung abnormality detection.
* 💻 **User-friendly Interface** — Built with **Streamlit** for easy image upload and real-time predictions.
* ⚙️ **High Accuracy** — Fine-tuned on chest X-ray datasets for robust performance.
* 📊 **Visualization Support** — Displays the uploaded X-ray for reference.
* 🔄 **Fast & Reliable** — Caches model for faster inference and smooth user experience.

---

## 🧬 Tech Stack

| Component                   | Technology                                |
| --------------------------- | ----------------------------------------- |
| **Frontend / UI**           | Streamlit                                 |
| **Backend**                 | Python                                    |
| **Deep Learning Framework** | TensorFlow / Keras                        |
| **Model Architecture**      | VGG19 (Transfer Learning)                 |
| **Libraries**               | NumPy, OpenCV, PIL                        |
| **Dataset**                 | Chest X-Ray Dataset (Normal vs Pneumonia) |

---

## 🏗️ Project Structure

```
CliniScan/
│
├── app.py                     # Streamlit web app
├── vgg_unfrozen.h5            # Trained model weights
├── Pneumonia Detection.ipynb  # Model training notebook
├── requirements.txt           # Required dependencies
└── README.md                  # Project documentation
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-username>/CliniScan.git
cd CliniScan
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Application

```bash
streamlit run app.py
```

### 4️⃣ Upload an Image

Upload a chest X-ray (in `.jpg`, `.jpeg`, or `.png` format) to see whether it’s **Normal** or **Pneumonia**.

---

## 🧠 Model Architecture

* **Base Model:** VGG19 (pre-trained on ImageNet)
* **Added Layers:**

  * Flatten Layer
  * Dense (ReLU) Layers
  * Dropout Layer (to prevent overfitting)
  * Output Layer (Softmax for binary classification)

The model takes a **128×128 RGB image** as input and outputs a class prediction:

* **0 → Normal**
* **1 → Pneumonia**

---

## 📈 Results

| Metric        | Value                             |
| ------------- | --------------------------------- |
| **Accuracy**  | ~95%                              |
| **Precision** | High                              |
| **Recall**    | Strong recall for Pneumonia cases |

*(Values may vary depending on dataset and hyperparameters)*

---

## 🧑‍⚕️ Applications

* Medical image screening support
* Hospital AI diagnostic assistants
* Early detection of pneumonia in rural clinics
* Radiology image classification research

---

## 💡 Future Enhancements

* 🩸 Add multi-class classification (e.g., COVID-19, Tuberculosis)
* 🌐 Deploy model on cloud (AWS/GCP/Streamlit Cloud)
* 📱 Develop a mobile-friendly interface
* 🔍 Integrate Grad-CAM for explainable AI visualizations

---

## 👨‍💻 Author

**Yemineni Jyothi Madhava Saikrishna**
🎓 B.Tech | Vasireddy Venkatadri Institute of Technology
💼 AI & ML Enthusiast | Intern at Microsoft AI & Azure, Infosys Springboard
📧 [saikrishnayjm2006@gmail.com](mailto:saikrishnayjm2006@gmail.com)

---

## 🏁 Acknowledgements

* Dataset by [Kaggle: Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
* TensorFlow & Keras open-source frameworks
* Streamlit for rapid AI web deployment

---

## 📜 License

This project is licensed under the **MIT License**.
You are free to use, modify, and distribute this work with proper attribution.

---

## 🌐 Live Demo

🚀 **Run Locally:** [http://localhost:8501](http://localhost:8501)

