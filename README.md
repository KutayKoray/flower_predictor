# Flower Predictor  

This project includes a deep learning model and a Flask-based web server that analyzes a flower image and predicts its species.  

## 📌 Features  
- **Transfer Learning**: Trained using the MobileNetV2 model.  
- **Data Processing**: Data augmentation techniques were applied to improve accuracy.  
- **Flask API**: A web interface allows users to upload a flower image and get predictions.  
- **Top 3 Predictions**: The model returns the top 3 flower species with their probability percentages.  

## 🚀 Installation  
Follow these steps to run the project.  

### 1️⃣ Clone the Repository  
```bash  
git clone git@github.com:KutayKoray/flower_predictor.git  
cd flower_predictor  
```  

### 2️⃣ Install Dependencies  
Install the required Python dependencies using:  
```bash  
pip install -r requirements.txt  
```  

### 3️⃣ Train the Model (Optional)  
If you want to train your own model:  
```bash  
python train.py  
```  
This will generate the `best_flower_model.h5` file.  

### 4️⃣ Start the Flask Server  
```bash  
python server.py  
```  
Once the server is running, you can access it at `http://127.0.0.1:5000/`.  

## 📸 Usage  
1. Open the web interface.  
2. Upload a flower image.  
3. View the model’s predictions and probabilities.  

Alternatively, you can get predictions via the API:  
```bash  
curl -X POST -F "file=@path/to/image.jpg" http://127.0.0.1:5000/predict  
```  

## 📂 Project Structure  
```
flower_predictor/  
│── flowers/                # Flower dataset  
│── static/uploads/         # Directory for uploaded images  
│── train.py                # Model training script  
│── server.py               # Flask server script  
│── best_flower_model.h5    # Pre-trained model  
│── requirements.txt        # List of dependencies  
│── README.md               # Project documentation  
```

## 📜 License  
This project is licensed under the MIT License.  

---  
📌 Feel free to contribute by submitting a PR! 🚀  

