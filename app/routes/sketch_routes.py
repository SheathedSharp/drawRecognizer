'''
Author: SheathedSharp z404878860@163.com
Date: 2025-01-12 12:44:16
'''
from flask import Blueprint, jsonify, request, send_file, render_template
from app.models.sketch_model import SketchRetriever
from config import Config
import os

sketch_bp = Blueprint('sketch', __name__)

# 初始化检索器并加载预训练模型（暂时不加载训练好的模型）
retriever = SketchRetriever()

# 图库路径指向你的图片目录
gallery_dir = os.path.join(Config.SKETCHY_PHOTO_DIR)
# 确保图库目录存在
if not os.path.exists(gallery_dir):
    print(f"Gallery directory not found: {gallery_dir}")
else:
    print(f"Building gallery from: {gallery_dir}")
    # 每个类别只选择10张图片
    retriever.build_gallery(gallery_dir, samples_per_category=10)

@sketch_bp.route('/api/retrieve', methods=['POST'])
def retrieve_similar():
    data = request.get_json()
    if 'sketch' not in data:
        return jsonify({'error': 'No sketch provided'}), 400
    
    # 获取前10个相似结果
    results = retriever.retrieve(data['sketch'], top_k=10)
    return jsonify({'results': results})

@sketch_bp.route('/sketch')
def sketch_page():
    return render_template('sketch.html')

@sketch_bp.route('/status')
def status():
    return jsonify({
        'status': 'running',
        'gallery_size': len(retriever.gallery_paths) if retriever.gallery_paths else 0
    })

@sketch_bp.route('/gallery_image/<path:filename>')
def gallery_image(filename):
    """提供图库图片访问"""
    try:
        # 构建完整的文件路径
        image_path = os.path.join(Config.SKETCHY_PHOTO_DIR, filename)
        return send_file(image_path)
    except Exception as e:
        print(f"Error serving image {filename}: {e}")
        return "Image not found", 404