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

def init_model_status():
    models_status = model_manager.check_models_status()
    if not os.environ.get('WERKZEUG_RUN_MAIN'):  # 只在主进程中打印
        print("\n=== Digit Recognition Models Status ===")
        for model_name, status in models_status.items():
            print(f"\nModel: {model_name}")
            print(f"- Status: {'Available' if status['exists'] else 'Not Found'}")
            print(f"- Type: {status['type']}")
            print(f"- Description: {status['description']}")
            if status.get('accuracy'):
                print(f"- Best Accuracy: {status['accuracy']}%")
    return models_status

def get_recognizer():
    default_model_name = 'cnn_c32c64_k5_fc512'
    model_config = Config.DIGIT_MODEL_CONFIGS[default_model_name]
    model_path = model_config['path']
    return DigitRecognizer(model_name=default_model_name, model_path=model_path)

# 初始化全局变量
model_manager = DigitModelManager()
models_status = init_model_status()
recognizer = get_recognizer()

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
    
@digit_bp.route('/api/models', methods=['GET'])
def get_models():
    models_info = {}
    for model_name in model_manager.models.keys():
        if model_name in model_manager.model_stats:
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