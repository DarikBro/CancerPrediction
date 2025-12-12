# CancerPrediction


# 🛠️ Installation & Setup Guide

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

Bash
python -m venv venv
.\venv\Scripts\activate

For Mac/Linux:
Bash

python3 -m venv venv
source venv/bin/activate
3️⃣ Install Dependencies
Install the required Python libraries using pip. This project relies on PyTorch, Streamlit, and Jupyter.

Bash

pip install torch torchvision numpy matplotlib streamlit jupyter
Note: If you have a dedicated GPU (NVIDIA), you may want to install the specific CUDA version of PyTorch for faster training. Visit pytorch.org for the correct command.

4️⃣ Generate the Model & Data (CRITICAL STEP)
⚠️ The model.pth file and data/ folder are NOT included in this download.

You must run the training script once to download the PCAM dataset  and train the model locally.

Launch Jupyter Notebook:

Bash

jupyter notebook
Open the file training.ipynb.

Click "Run All" (or run cells sequentially).

What this does:

Downloads the PatchCamelyon dataset to a ./data folder.

Trains the CNN for 20 epochs.

Saves the trained weights to pathonet_model.pth (or model.pth).

5️⃣ Run the Application
Once the training is finished and you see the .pth model file in your folder, you can launch the web interface.

Bash

streamlit run app.py
A local web server will start (typically at http://localhost:8501).

You can now upload histopathology images to test the model .
