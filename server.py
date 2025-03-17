from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import tensorflow as tf
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model = tf.keras.models.load_model('best_flower_model.h5')

class_indices = {
    "0": "bellflower",
    "1": "daisy",
    "2": "dandelione",
    "3": "lotus",
    "4": "rose",
    "5": "sunflower",
    "6": "tulip"
}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Dosya bulunamadı'})
        
    file = request.files['file']
        
    if file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi'})
        
    if file and allowed_file(file.filename):
        file_ext = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4().hex}{file_ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
                
        img = tf.keras.preprocessing.image.load_img(filepath, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
                
        predictions = model.predict(img_array)
                
        top_3_indices = predictions[0].argsort()[-3:][::-1]
        top_3_predictions = []
        for i in top_3_indices:
            top_3_predictions.append({
                'class': class_indices[str(i)],
                'probability': float(predictions[0][i] * 100)
            })
                
        return jsonify({
            'success': True,
            'filename': unique_filename,
            'predictions': top_3_predictions
        })
        
    return jsonify({'error': 'İzin verilen dosya türleri: png, jpg, jpeg'})

if __name__ == '__main__':
    app.run(debug=True)