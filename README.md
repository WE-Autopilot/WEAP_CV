### **1️⃣ Ensure `.gitignore` is Correct**
Since your **`venv/` folder** is inside the project, make sure it's ignored in `.gitignore`:
```plaintext
# Ignore virtual environment
venv/

# Ignore dataset files (if downloading separately)
datasets/

# Ignore Jupyter Notebook checkpoints
.ipynb_checkpoints/

# Ignore logs and results
logs/
results/

# Ignore compiled Python files
__pycache__/
*.pyc
*.pyo
*.pyd

# Ignore model checkpoints and large files
*.pt
*.pth
*.onnx
models/*.pth

# Ignore system files
.DS_Store
Thumbs.db
```

---

### **2️⃣ Verify `requirements.txt` is Complete**
Your dependencies should be installed with:
```bash
pip install -r requirements.txt
```
To **update** `requirements.txt` if you installed new packages:
```bash
pip freeze > requirements.txt
```

---

### **3️⃣ Check Kaggle Setup**
You have a **datasets/** folder, but it looks empty. Using Kaggle download the datasets:
1. **Install Kaggle API**:
   ```bash
   pip install kaggle
   ```
2. **Move your `kaggle.json` API key** to:
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```
3. **Download datasets** (example for COCO & KITTI):
   ```bash
   kaggle datasets download -d microsoft/coco
   unzip coco.zip -d datasets/coco/

   kaggle datasets download -d kitti/kitti-object-detection
   unzip kitti-object-detection.zip -d datasets/kitti/
   ```
4. **Verify dataset files exist**:
   ```bash
   ls datasets/coco/
   ls datasets/kitti/
   ```

---

### **4️⃣ Test Training & Inference Scripts**
Run **quick tests** to make sure everything works properly.

#### ✅ **Check if PyTorch is installed correctly**:
```bash
python -c "import torch; print(torch.__version__)"
```

#### ✅ **Test YOLOv8 Training**
```bash
python train.py --model yolov8 --dataset coco
```

#### ✅ **Test DeepSORT Tracking**
```bash
python inference.py --model deepsort --input sample_video.mp4
```

#### ✅ **Test Lane Detection**
```bash
python inference.py --model lane_detection --input road_image.jpg
```

---

### **5️⃣ Run Jupyter Notebooks**
Install Jupyter if not installed:
```bash
pip install jupyter notebook
jupyter notebook
```
Then open and verify:
- `notebooks/deepsort_tracking.ipynb`
- `notebooks/lane_detection.ipynb`
- `notebooks/yolov8_training.ipynb`

---
