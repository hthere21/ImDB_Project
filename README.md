# ImDB_Project

## Overview

This project focuses on sentiment analysis using machine learning techniques. It aims to classify movie reviews into positive and negative sentiments based on their textual content. The implementation uses natural language processing (NLP) tools, machine learning algorithms, and various text vectorization methods.

## Getting Started

### Prerequisites

- Python 3.8
- Required Python packages: nltk, scikit-learn, matplotlib, numpy

Install the necessary packages using:
- pip install nltk scikit-learn matplotlib numpy

### 1. Installation
Clone the repository:
- git clone https://github.com/your-username/sentiment-analysis-project.git
- cd sentiment-analysis-project

### 2. Download NLTK data:
- python -m nltk.downloader punkt

### 3. Run the project:
- python main.py
Project Structure
  main.py: Main script to execute the sentiment analysis tasks.
  data_preprocessing.py: Contains functions for loading and preprocessing the dataset.
  text_vectorization.py: Implements text vectorization using different methods (CountVectorizer, TF-IDF).
  model_training.py: Defines and trains the sentiment analysis model using MLPClassifier.
  evaluate_model.py: Evaluates the trained model and prints accuracy.
  example_prediction.py: Demonstrates how to make predictions on new sentences.

### Usage
Adjust the file paths in main.py to point to your training and testing datasets.
Copy code
train_path = '/path/to/train'
test_path = '/path/to/test'
Run main.py to execute the sentiment analysis pipeline.
bash
Copy code
python main.py
Explore the results and model performance in the console output.
### 4. Results
The project achieves the following results:

- Accuracy using CountVectorizer with 1-gram: 83.94%
- Accuracy using CountVectorizer with 2-grams: 75.74%
- Accuracy using CountVectorizer with 3-grams: 64.54%
- Accuracy using CountVectorizer with 1-grams and 2-grams: 84.3%
- Accuracy using CountVectorizer with 1-grams, 2-grams, and 3-grams: 84.1%
- Accuracy using TF-IDF with 1-grams: 83.67%
