'''
Author: SheathedSharp z404878860@163.com
Date: 2025-01-12 12:41:00
'''
from flask import Flask
from config import Config

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Register blueprints
    from app.routes.digit_routes import digit_bp
    from app.routes.sketch_routes import sketch_bp
    
    app.register_blueprint(digit_bp)
    app.register_blueprint(sketch_bp)

    return app