# Speech Detection with LSTM and Spectrogram

This repository contains a Python script for speech detection using Long Short-Term Memory (LSTM) neural networks and spectrogram features. The code is designed to classify spoken words or sounds into predefined categories.

## Prerequisites

Before you begin, ensure you have the following dependencies installed:

- Python (>=3.6)
- NumPy
- pandas
- librosa
- scikit-learn
- TensorFlow (>=2.0)
- tqdm

You can install these dependencies using `pip`:

```bash
pip install numpy pandas librosa scikit-learn tensorflow tqdm

Getting Started
Clone this repository to your local machine:
bash

git clone https://github.com/yourusername/speech-detection-lstm.git
cd speech-detection-lstm
Download the dataset (recordings) and unzip it into the recordings/ directory.
bash

unzip recordings.zip -d recordings/
Usage
The main script for speech detection is speech_detection.py.
bash

python speech_detection.py
The script performs the following steps:
Data preprocessing
Data augmentation
Model training (LSTM with spectrogram features)
Model evaluation
You can customize the script for your specific dataset or requirements.

Data Preparation
Ensure your dataset is organized in the recordings/ directory.
Audio files should be named like label_username_index.wav, where label represents the class label, username is an identifier, and index is the file index.
Augmentation
The script includes data augmentation by changing the time stretch and pitch of audio samples to increase the size of the training dataset.
Model
The model architecture consists of an LSTM layer followed by a few fully connected layers for classification.
You can adjust the model architecture and hyperparameters in the script to optimize performance.
Evaluation
The script evaluates the model's performance using F1-score as a metric.
It also includes a step to convert audio data into spectrogram representations for input to the LSTM model.
Contributing
Contributions are welcome! If you find any issues or have ideas for improvements, please open an issue or create a pull request.