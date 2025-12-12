# Metastatic Tissue Detection using Deep Learning

**Author:** Ildar Mamin
**Course:** INFO-6147 Deep Learning with PyTorch
**Date:** December 2025

## Project Overview
This project implements a Deep Learning model to automatically detect metastatic cancer in histopathology images. It uses a Convolutional Neural Network (CNN) built with PyTorch to classify lymph node tissue as either **Healthy** or **Tumor**.

The project also includes a web application built with **Streamlit**, allowing users to upload tissue scans and receive real-time predictions.

> **Note:** The trained model file (`model.pth`) and the dataset are **not included** in this repository due to their large size. You must run the training notebook locally to download the data and generate the model file before running the app.

## Project Structure
* `training.ipynb`: The main Jupyter Notebook. Running this **downloads the dataset** and **trains the model**.
* `app.py`: The Streamlit web application for testing the model.
* `model.pth`: (Generated locally) The saved model weights.
* `data/`: (Generated locally) The folder containing the PCAM dataset.

## Installation & Setup Guide

This project requires **Python 3.8+** and several Deep Learning libraries. Because the dataset and trained model are large, they are not included in the repository and must be generated locally.

Follow these steps to set up the environment and run the application.

---

### 1️⃣ Clone the Repository
First, download the project code to your local machine.
```bash
git clone [https://github.com/your-username/metastasis-detection-project.git](https://github.com/your-username/metastasis-detection-project.git)
cd metastasis-detection-project
```

### 2️⃣ Set Up a Virtual Environment (Recommended)
It is best practice to create a virtual environment to keep dependencies organized.

For Windows:
```bash
python -m venv venv
.\venv\Scripts\activate
```

For Mac/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

###3️⃣ Install Dependencies
Install the required Python libraries using pip. This project relies on PyTorch, Streamlit, and Jupyter.

```bash
pip install torch torchvision numpy matplotlib streamlit jupyter
```

Note: If you have a dedicated GPU (NVIDIA), you may want to install the specific CUDA version of PyTorch for faster training. Visit pytorch.org for the correct command.

###4️⃣ Generate the Model & Data (CRITICAL STEP)
⚠️ The model.pth file and data/ folder are NOT included in this download.

You must run the training script once to download the PCAM dataset  and train the model locally.

Launch Jupyter Notebook:

```bash
jupyter notebook
```

1. Open the file training.ipynb.
2. Click "Run All" (or run cells sequentially).

What this does:
1. Downloads the PatchCamelyon dataset to a ./data folder.
2. Trains the CNN for 20 epochs.
3. Saves the trained weights to pathonet_model.pth (or model.pth).

###5️⃣ Run the Application
Once the training is finished and you see the .pth model file in your folder, you can launch the web interface.

```bash
streamlit run app.py
```

A local web server will start (typically at http://localhost:8501).

You can now upload histopathology images to test the model .
