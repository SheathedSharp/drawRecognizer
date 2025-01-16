'''
Author: SheathedSharp z404878860@163.com
Date: 2025-01-12 12:44:16
'''
from flask import Blueprint, jsonify, request, send_file, render_template
from app.models.sketch_model import SketchRetriever
import os

sketch_bp = Blueprint('sketch', __name__)
# retriever = SketchRetriever()

# # 初始化时构建图库
# gallery_dir = 'app/static/gallery'
# retriever.build_gallery(gallery_dir)

@sketch_bp.route('/api/retrieve', methods=['POST'])

@sketch_bp.route('/sketch')
def sketch_page():
    return render_template('sketch.html')


def retrieve_similar():
    data = request.get_json()
    if 'sketch' not in data:
        return jsonify({'error': 'No sketch provided'}), 400
    
    results = retriever.retrieve(data['sketch'])
    return jsonify({'results': results})