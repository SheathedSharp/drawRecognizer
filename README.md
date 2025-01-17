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
3. Run `python -m app.train.train_digit` to train the digit models, you can specify the models to train, e.g.: `python -m app.train.train_digit --model=cnn_c16c32_k3_fc10 mlp_512_256_128`, you can just run `python -m app.train.train_digit --model=DigitCNN_C32C64_K5_FC512` to train best model because `DigitCNN_C16C32_K3_FC10` is too simple to train(network is too simple) and `DigitCNN_C32C64C128_K3_FC256` is too complex to train(there aren't enough data to train).
4. Run `python download_skecthy_dataset.py` to download the sketch dataset
5. Run `unzip data/Sketchy/Sketchy.zip -d data/Sketchy/ && rm data/Sketchy/Sketchy.zip` to unzip the sketch dataset and remove the zip file
7. Run `python -m app.train.train_sketch` to train the sketch models(AutoDL GPU Server-RTX 4090 to train, aliyun OSS to translate data to AutoDL GPU Server)
8. Run the application: `python run.py`

## Project Structure
- /app: Main application directory
- /app/routes: API endpoints and route handlers
- /app/models: PyTorch models for digit recognition and sketch retrieval
- /app/static: Static files (CSS, JavaScript, images)
- /app/templates: HTML templates

## Usage
1. Access the digit recognition page at: http://localhost:5000/digit
2. Access the sketch retrieval page at: http://localhost:5000/sketch