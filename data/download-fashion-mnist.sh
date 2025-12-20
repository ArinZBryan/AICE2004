#!/bin/bash

f="./fashionmnist.zip"

echo "Downloading Fashion-MNIST..."

# Download the zip folder
curl -L -o ./fashionmnist.zip https://www.kaggle.com/api/v1/datasets/download/zalando-research/fashionmnist

# Extract only the CSV
unzip -o fashionmnist.zip '*.csv'

echo "Downloading Fashion-MNIST completed."
