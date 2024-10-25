# Real_time_translation

This project implements a multi-language translation application using LSTM models for translating English to Hindi, French, and Tamil. The application provides a user-friendly interface for entering text and selecting target languages for translation.

## Features

- Translates English text to Hindi, French, and Tamil.
- Supports translation of Hindi and French simultaneously.
- Checks for specific conditions before allowing translations (e.g., input length, starting with vowels).
- Handles errors and suggests similar words for unavailable entries.

## Requirements

### Packages

To run this project, ensure that the following Python packages are installed:

- **pandas**: Used for data manipulation and analysis.
- **numpy**: Used for numerical computations.
- **tensorflow**: The backend library for Keras; used for building and training deep learning models.
- **keras**: High-level neural networks API for building LSTM models.
- **tkinter**: Standard GUI toolkit for creating the application interface.
- **speech_recognition**: Optional library for speech recognition capabilities (not used in the current implementation).

### Package Versions

For compatibility, the following package versions are recommended:

pandas==1.5.3
numpy==1.21.6
tensorflow==2.12.0
keras==2.12.0 

### Dataset
The application uses the following CSV datasets for training and translation:

- **english_converted.csv** : Contains English sentences.
- **hindi_converted.csv** : Contains corresponding Hindi translations.
- **french_converted.csv** : Contains French translations.
- **tamil_converted.csv** : Contains Tamil translations.
Make sure these datasets are formatted correctly, with a column named text for sentences.
