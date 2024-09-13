# Cyberbullying Text Classifier

## Overview

This project involves creating a text classifier to identify types of cyberbullying in tweets. The classifier is built using a Recurrent Neural Network (RNN) with Bidirectional LSTM layers and is deployed with a PyQt5 GUI for user interaction.

## Project Structure

1. **Data Preprocessing**:
   - `cyberbullying_tweets.csv`: Original dataset containing tweets and their cyberbullying labels.
   - The preprocessing steps involve cleaning the text (removing URLs, punctuation, and numbers), tokenizing, removing stopwords, stemming, and lemmatizing.
   - Processed data is saved to `processed_cyberbullying_tweets.csv`.

2. **Model Training**:
   - The model is trained using a Bidirectional LSTM architecture.
   - The dataset is split into training and testing sets, and the model is evaluated using accuracy, precision, recall, and F1 score.
   - The trained model and tokenizer are saved as `cyberbullying_classifier_model.h5` and `tokenizer.pickle`, respectively.

3. **GUI Application**:
   - A PyQt5 application allows users to input text and receive predictions and class probabilities from the classifier.

## Modules Used

- **Pandas**: Data manipulation and analysis.
- **NumPy**: Numerical computations.
- **Re**: Regular expressions for text cleaning.
- **NLTK**: Natural Language Toolkit for tokenization, stopword removal, stemming, and lemmatization.
- **WordCloud**: Generation of word clouds for visualizing word frequency.
- **Matplotlib**: Plotting library for visualizing word clouds and confusion matrices.
- **Keras**: Deep learning library for building and training the text classifier model.
- **Scikit-learn**: Evaluation metrics and data splitting.
- **Seaborn**: Visualization library for confusion matrices.
- **PyQt5**: GUI library for creating the text classification application.
- **Pickle**: Serialization of model and tokenizer.

## How to Run

1. **Install Dependencies**: Ensure all required modules are installed using `pip install -r requirements.txt`.

2. **Preprocess Data**: Run the preprocessing script to clean and save the data.

3. **Train Model**: Execute the training script to build and save the model.

4. **Run GUI Application**:
   - Execute the GUI script to open the application.
   - Enter text to classify and view predictions and probabilities.

## Files

- `cyberbullying_tweets.csv`: Original dataset.
- `cyberbullying_tweets1.csv`: Preprocessed dataset.
- `cyberbullying_classifier_model.h5`: Trained model.
- `tokenizer.pickle`: Tokenizer for text preprocessing.
- `model.py`: Script for defining and training the model.
- `recognitions.py`: PyQt5 application for user interaction.
