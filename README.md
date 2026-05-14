# Intelligent Risk Platform

Intelligent web system for risk monitoring and geospatial visualization based on neural network analysis of video data.

## Project Overview

This project combines computer vision, neural networks and geospatial analytics in a single web platform for monitoring risk situations.

The system analyzes video data using a trained EfficientNet-B0 neural network and visualizes statistical risk indicators on an interactive world map.

## Features

- Video risk analysis
- Neural network classification
- Interactive risk map
- Time-based analytics
- Dashboard visualization
- Geospatial data analysis
- Real-time alerts

## Technologies Used

- Python
- Django
- PyTorch
- OpenCV
- Pandas
- Plotly
- HTML/CSS

## Neural Network

The project uses the EfficientNet-B0 model trained to classify video frames into three categories:

- normal
- suspicious
- high-risk

## Dashboard Components

The dashboard includes:

- Interactive world risk map
- Surveillance video feed
- AI detection panel
- Timeline analytics
- Real-time alerts

## Dataset

The project uses statistical data from:

- UNODC
- GLOTIP dataset

## Running the Project

Install dependencies:
pip install django pandas plotly torch torchvision pillow openpyxl opencv-python

Run the Django server:
python manage.py runserver

Open in browser:
http://127.0.0.1:8000

Author - Khrystyna Kosiv

```bash
pip install django pandas plotly torch torchvision pillow openpyxl opencv-python
