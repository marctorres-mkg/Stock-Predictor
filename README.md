# Stock-Predictor

A personal deep learning project for predicting stock movements from historical market data using PyTorch.

## Overview

Stock-Predictor is a personal project exploring neural networks for stock movement prediction using historical financial time-series data.

The model uses daily market data from 7 technology companies:

- AAPL
- AMD
- AMZN
- INTC
- META
- MSFT
- NVDA

The dataset covers historical data from 2015 to 2026 and includes:

- Open
- High
- Low
- Close
- Volume

## Data Preparation

The data is transformed before being used for training.

For Open, High, Low, and Close, logarithmic returns are calculated. Volume is transformed using a logarithmic transformation and subsequently normalized.

The data is then organized into **3-day windows** and flattened before being passed to the neural network.

The model predicts whether the next movement is positive for each of the 7 stocks.

## Neural Network

The neural network is implemented using PyTorch tensor operations.

It contains:

- An input layer
- Two hidden layers with 64 neurons each
- ReLU activation functions
- An output layer with 7 outputs
- Sigmoid activation for predictions

The model parameters are optimized using gradient-based updates and backpropagation.

## Training

The dataset is split into training and validation sets using an 80/20 split.

The current training configuration uses:

- Learning rate: `0.01`
- Epochs: `100`
- Hidden layers: `[64, 64]`

The trained model parameters are saved to `model.pkl`.

## Results

The model achieved approximately **52% accuracy** on the prediction task.

This project was developed as a practical exploration of:

- Neural networks
- PyTorch
- Time-series data processing
- Feature engineering
- Data normalization
- Backpropagation
- Gradient-based optimization
- Tensor manipulation

## Project Structure

```text
Stock-Predictor/
│
├── README.md
├── requirements.txt
├── functions.py
├── training.py
└── testing.py
```

## Installation

Install the required dependencies with:

```bash
pip install -r requirements.txt
```

## Training

Run:

```bash
python training.py
```

This downloads the historical market data, prepares the dataset, trains the neural network, evaluates its accuracy, and saves the model parameters to `model.pkl`.

## Testing

Run:

```bash
python testing.py
```

This downloads recent market data, loads the trained model, and generates predictions for the 7 stocks.

## Disclaimer

This is a personal educational project and is not intended to provide financial advice or reliable investment predictions.
