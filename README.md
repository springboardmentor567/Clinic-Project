# 🫁 CliniScan: Lung pneumonia Detection

<div align="center">

![CliniScan Banner](https://img.shields.io/badge/AI-Powered-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-green?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

**An intelligent system for automated detection and localization of lung abnormalities from chest X-ray images**

[🚀 Live Demo](https://cliniscan-lung-abnormality-detection-on-chest-x-rays-gp9fjubya.streamlit.app/) • [📖 Documentation](#features) • [🐛 Report Bug](https://github.com/Jasmin-IT/CliniScan-Lung-Abnormality-Detection-on-Chest-X-rays/issues)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Demo](#demo)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Model Information](#model-information)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## 🔍 Overview

CliniScan leverages advanced deep learning techniques to automatically analyze chest X-ray images and detect potential lung abnormalities. Trained on the comprehensive VinDr-CXR dataset, our system can identify various pulmonary conditions including:

- 🔴 **Opacities** - Cloudy areas in lung tissue
- 🟡 **Fibrosis** - Scarring of lung tissue
- 🟢 **Other abnormalities** - Various lung pathologies

**Target Audience:** Radiologists, medical professionals, and healthcare institutions looking to enhance diagnostic accuracy and efficiency.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🤖 **AI-Powered Analysis** | Deep learning model trained on medical imaging datasets |
| 📍 **Abnormality Localization** | Precise detection and marking of affected regions |
| 🖼️ **DICOM Support** | Compatible with standard medical imaging formats |
| ⚡ **Real-time Processing** | Fast inference for immediate clinical feedback |
| 🌐 **Web Interface** | User-friendly Streamlit application |
| 📊 **Visual Reports** | Clear visualization of detected abnormalities |

---

## 🎥 Demo

### Try it Live!

Experience CliniScan in action: **[Launch Live Demo](https://cliniscan-lung-abnormality-detection-on-chest-x-rays-gp9fjubya.streamlit.app/)**

### Screenshots

> *Upload a chest X-ray image and receive instant AI-powered analysis with visual annotations*

---

## 🔧 Prerequisites

Before getting started, ensure you have the following installed:

- **Python 3.8 or higher**
- **pip** (Python package manager)
- **Git** (for cloning the repository)

### Required Python Libraries

```
streamlit
torch
torchvision
pydicom
opencv-python-headless
pandas
```

> 💡 **Note:** All dependencies are listed in `requirements.txt` for easy installation.

---

## 📥 Installation

Follow these simple steps to set up CliniScan on your local machine:

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Jasmin-IT/CliniScan-Lung-Abnormality-Detection-on-Chest-X-rays.git
cd CliniScan-Lung-Abnormality-Detection-on-Chest-X-rays
```

### 2️⃣ Create a Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Verify Installation

```bash
python --version
pip list
```

---

## 🚀 Usage

### Running the Application

Start the Streamlit web application with a single command:

```bash
streamlit run app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`

### Using the Interface

1. **Upload Image**: Click the upload button and select a chest X-ray image (PNG, JPG, JPEG, or DICOM format)
2. **Process**: The AI model will automatically analyze the image
3. **Review Results**: View detected abnormalities with visual annotations
4. **Download Report**: Export results for clinical documentation

### Supported Image Formats

- 📄 **DICOM** (.dcm)
- 🖼️ **PNG** (.png)
- 📷 **JPEG** (.jpg, .jpeg)

---

## 🧠 Model Information

### Architecture

CliniScan utilizes a state-of-the-art convolutional neural network optimized for medical image analysis.

### Training Dataset

**VinDr-CXR Dataset**
- Large-scale chest X-ray dataset
- Annotated by experienced radiologists
- Diverse range of pulmonary conditions

### Performance Metrics

| Metric | Score |
|--------|-------|
| Accuracy | TBD |
| Sensitivity | TBD |
| Specificity | TBD |

> ⚠️ **Disclaimer:** CliniScan is designed to assist medical professionals, not replace them. Always consult qualified healthcare providers for medical decisions.

---

## 📁 Project Structure

```
CliniScan/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── model/                  # Trained model files
│   └── weights.pth
├── utils/                  # Utility functions
│   ├── preprocessing.py
│   └── visualization.py
├── data/                   # Sample data (not tracked)
└── README.md              # This file
```

---

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

1. 🍴 **Fork** the repository
2. 🌿 **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. 💾 **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. 📤 **Push** to the branch (`git push origin feature/AmazingFeature`)
5. 🔀 **Open** a Pull Request

### Code of Conduct

Please be respectful and constructive in all interactions with the project community.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

**Project Maintainer:** Jasmin-IT

- 🐙 GitHub: [@Jasmin-IT](https://github.com/Jasmin-IT)
- 📬 Issues: [Report a bug or request a feature](https://github.com/Jasmin-IT/CliniScan-Lung-Abnormality-Detection-on-Chest-X-rays/issues)

---

## 🙏 Acknowledgments

- VinDr-CXR dataset creators and contributors
- Open-source medical imaging community
- PyTorch and Streamlit development teams

---

<div align="center">

**Made with ❤️ for better healthcare outcomes**

⭐ **Star this repository if you find it helpful!** ⭐

</div>
