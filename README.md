# Cryptocurrency Price Prediction #

This application predicts the next 3 days’ Bitcoin close prices by training a NeuralProphet model on the past 3 days’ sentiment scores (scraped via snscrape) and close‑price history. Because Twitter began requiring login in mid‑April 2023—breaking unauthenticated scraping—our sentiment data stopped on April 17 2023, so predictions cover only April 18–20.

### <ins>Python version</ins> ###
Python 3.11

### <ins>Dependencies</ins> ###
A list of Python packages needs to be installed.

### <ins>Steps to run the script</ins> ###
1. Create a new virtual environment:
```bash
conda create -n env python=3.11
conda activate env
```
2. Install the required packages:
```bash
pip install -r "/Users/path/to/requirements.txt"
```
3. Set the Keras backend to TensorFlow:
```bash
export KERAS_BACKEND=tensorflow
```
4. Launch the application:
```bash
streamlit run Main.py
```
