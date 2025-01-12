'''
Author: SheathedSharp z404878860@163.com
Date: 2025-01-12 12:43:52
'''
from flask import Blueprint, render_template, jsonify, request
# from app.models.digit_model import DigitRecognizer

digit_bp = Blueprint('digit', __name__)
# digit_model = DigitRecognizer()

@digit_bp.route('/digit')
def digit_page():
    return render_template('digit.html')

@digit_bp.route('/api/recognize', methods=['POST'])
def recognize_digit():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    # Add recognition logic here
    return jsonify({'result': 'Recognition result'})