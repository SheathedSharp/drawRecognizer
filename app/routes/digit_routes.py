'''
Author: SheathedSharp z404878860@163.com
Date: 2025-01-12 12:43:52
'''
from flask import Blueprint, render_template, jsonify, request
from app.models.digit_model import DigitRecognizer
from app.models.model_manager import DigitModelManager
from config import Config
import os

digit_bp = Blueprint('digit', __name__)
model_manager = DigitModelManager()

# 检查所有模型的状态
models_status = model_manager.check_models_status()
print("\n=== Digit Recognition Models Status ===")
for model_name, status in models_status.items():
    print(f"\nModel: {model_name}")
    print(f"- Status: {'Available' if status['exists'] else 'Not Found'}")
    print(f"- Type: {status['type']}")
    print(f"- Description: {status['description']}")
    if status.get('accuracy'):
        print(f"- Best Accuracy: {status['accuracy']}%")

# 默认使用手写数字第一个模型
default_model_name = list(Config.DIGIT_MODEL_CONFIGS.keys())[0]
model_config = Config.DIGIT_MODEL_CONFIGS[default_model_name]
model_path = model_config['path']


recognizer = DigitRecognizer(model_path=model_path)

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
    
    
    
@digit_bp.route('/api/models', methods=['GET'])
def get_models():
    models_info = {}
    for model_name in model_manager.models.keys():
        model_info = model_manager.get_model_info(model_name)
        models_info[model_name] = {
            'description': model_info['description'],
            'type': model_info['type'],
            'parameters': model_info['parameters'],
            'stats': model_manager.model_stats.get(model_name, {})
        }
    return jsonify(models_info)

@digit_bp.route('/api/switch_model', methods=['POST'])
def switch_model():
    data = request.get_json()
    model_name = data.get('model_name')
    
    if model_name not in model_manager.models:
        return jsonify({'error': 'Invalid model name'}), 400
        
    model_path = Config.DIGIT_MODEL_CONFIGS[model_name]['path']
    if not os.path.exists(model_path):
        return jsonify({'error': 'Model file not found'}), 404
        
    global recognizer
    recognizer = DigitRecognizer(model_name=model_name, model_path=model_path)
    return jsonify({'message': f'Switched to model: {model_name}'})