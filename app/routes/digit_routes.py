'''
Author: SheathedSharp z404878860@163.com
Date: 2025-01-12 12:43:52
'''
from flask import Blueprint, render_template, jsonify, request
from app.models.digit_model import DigitRecognizer
import os

digit_bp = Blueprint('digit', __name__)
model_path = 'app/models/digit_model.pth'

# 如果模型文件不存在，训练新模型
if not os.path.exists(model_path):
    print("Training new model...")
    recognizer = DigitRecognizer()
    recognizer.train(epochs=5, save_path=model_path)
else:
    print("Loading existing model...")
    recognizer = DigitRecognizer(model_path)

@digit_bp.route('/digit')
def digit_page():
    return render_template('digit.html')

@digit_bp.route('/api/recognize', methods=['POST'])
def recognize_digit():
    data = request.get_json()
    if 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400
    
    result = recognizer.predict(data['image'])
    if result is None:
        return jsonify({'error': 'Recognition failed'}), 400
    
    return jsonify({'result': str(result)})

@digit_bp.route('/api/train', methods=['POST'])
def train_model():
    try:
        recognizer.train(epochs=5, save_path=model_path)
        return jsonify({'message': 'Training completed successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500