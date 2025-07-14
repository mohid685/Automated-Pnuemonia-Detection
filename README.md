# Automated Pneumonia Detection System

This project is a deep learning-based system designed to automatically detect pneumonia from chest X-ray images. It leverages convolutional neural networks (CNNs) to classify input radiographs as either normal or pneumonia-affected. The system is intended for educational and research purposes to demonstrate how machine learning can assist in medical diagnosis.

## Features

- Binary classification: Pneumonia vs Normal
- Trained on publicly available chest X-ray datasets
- Real-time image prediction support
- Model performance evaluation with accuracy, precision, recall, and confusion matrix
- Modular codebase with separation of concerns (data processing, training, evaluation)

## Project Structure

```
pneumonia-detection/
├── data/                  # Dataset directory
│   ├── train/
│   ├── test/
│   └── val/
├── models/                # Saved trained models
├── notebooks/             # Jupyter notebooks for EDA and experiments
├── src/                   # Core source code
│   ├── dataloader.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── utils/                 # Helper functions and visualization
├── requirements.txt       # Project dependencies
├── README.md              # Project documentation
└── main.py                # Entry point for training or prediction
```

## Dataset

The model is trained using the **Chest X-Ray Images (Pneumonia)** dataset, originally published by Kermany et al. and made available by Kaggle:
- [Dataset Link](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

Dataset consists of:
- 5,000+ chest X-ray images
- Two classes: NORMAL and PNEUMONIA

## How to Use

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/pneumonia-detection.git
cd pneumonia-detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model

```bash
python src/train.py --epochs 10 --batch_size 32
```

### 4. Evaluate the Model

```bash
python src/evaluate.py --model_path models/best_model.pth
```

### 5. Predict on New Image

```bash
python src/predict.py --image_path path/to/image.jpg
```

## Model Architecture

- CNN-based model with multiple convolutional and pooling layers
- Dropout and batch normalization used for regularization
- Output layer with sigmoid activation for binary classification

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
