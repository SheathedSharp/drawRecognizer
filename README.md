<!--
 * @Author: SheathedSharp z404878860@163.com
 * @Date: 2025-01-12 12:34:38
-->
# Digit Recognition and Sketch Retrieval Web Application

This is a Flask-based web application that provides two main functionalities:
1. Handwritten Digit Recognition
2. Sketch Image Retrieval

## Features
- Real-time handwritten digit recognition using PyTorch
- Sketch-based image retrieval system
- Web-based drawing interface
- RESTful API endpoints

## Installation
1. Clone the repository: `git clone https://github.com/...`
2. Install dependencies: `pip install -r requirements.txt` or `conda env create -f environment.yml`
3. Run `python -m app.train.train_digit` to train the digit models, you can specify the models to train, e.g.: `python -m app.train.train_digit cnn_c16c32_k3_fc10 mlp_512_256_128`
4. Run `python -m app.train.train_sketch` to train the sketch models
4. Run the application: `python run.py`

## Project Structure
- /app: Main application directory
- /app/routes: API endpoints and route handlers
- /app/models: PyTorch models for digit recognition and sketch retrieval
- /app/static: Static files (CSS, JavaScript, images)
- /app/templates: HTML templates

## Usage
1. Access the digit recognition page at: http://localhost:5000/digit
2. Access the sketch retrieval page at: http://localhost:5000/sketch