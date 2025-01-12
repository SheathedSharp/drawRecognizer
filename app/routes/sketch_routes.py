'''
Author: SheathedSharp z404878860@163.com
Date: 2025-01-12 12:44:16
'''
from flask import Blueprint, render_template, jsonify, request
# from app.models.sketch_model import SketchRetriever

sketch_bp = Blueprint('sketch', __name__)
# sketch_model = SketchRetriever()

@sketch_bp.route('/sketch')
def sketch_page():
    return render_template('sketch.html')

@sketch_bp.route('/api/retrieve', methods=['POST'])
def retrieve_sketch():
    if 'sketch' not in request.files:
        return jsonify({'error': 'No sketch provided'}), 400
    
    # Add retrieval logic here
    return jsonify({'results': 'Retrieval results'})