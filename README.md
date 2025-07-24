# House Price Prediction

A machine learning application that predicts house prices based on various features like median income, house age, location, etc. The model is trained on the Housing dataset and served via a Streamlit web interface.

## Features

- Linear Regression model trained on California Housing dataset
- Interactive web interface with sliders for input features
- Real-time price prediction
- Simple deployment with Streamlit

## Prerequisites

- Python 3.7+
- pip (Python package manager)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/house-price-prediction.git
cd house-price-prediction

2.
```bash
python -m venv venv
source venv\Scripts\activate

3.
```bash
pip install -r requirements.txt
python model.py
streamlit run app.py