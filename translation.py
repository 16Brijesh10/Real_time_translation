import pandas as pd
import numpy as np
import json
import datetime
import tkinter as tk
from tkinter import ttk, messagebox
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
from keras.layers import Input, LSTM, Embedding, Dense
from keras.utils import to_categorical
import speech_recognition as sr

# Define constants
MAX_LENGTH = 10  # For English to Hindi and French
EMBEDDING_DIM = 256
LATENT_DIM = 256

# Initialize error tracking
wrong_words = []

# Load the dataset
def load_dataset(filename):
    return pd.read_csv(filename)

# Load Tokenizer
def load_tokenizer(filename):
    from keras.preprocessing.text import tokenizer_from_json
    with open(filename) as f:
        return tokenizer_from_json(json.load(f))

# Decode sequences
def decode_sequence(sequence, tokenizer):
    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
    decoded_sentence = []
    for index in sequence:
        if index > 0:
            word = reverse_word_map[index]
            decoded_sentence.append(word)
    return ' '.join(decoded_sentence)

# Build the LSTM model
def build_model(input_dim, output_dim):
    encoder_inputs = Input(shape=(None,))
    encoder_embedding = Embedding(input_dim=input_dim, output_dim=EMBEDDING_DIM)(encoder_inputs)
    encoder_lstm = LSTM(LATENT_DIM, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None,))
    decoder_embedding = Embedding(input_dim=output_dim, output_dim=EMBEDDING_DIM)(decoder_inputs)
    decoder_lstm = LSTM(LATENT_DIM, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    decoder_dense = Dense(output_dim, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Translate English to Hindi
def translate_english_to_hindi(english_sentence, english_tokenizer, english_to_hindi_model):
    if english_sentence[0].lower() in 'aeiou':
        raise ValueError("This word starts with a vowel. Provide some other words.")
    
    current_time = datetime.datetime.now()
    if not (current_time.hour == 21):  # around 9 PM
        raise ValueError("This model is available for words starting with vowels only between 9 PM and 10 PM.")

    sequence = english_tokenizer.texts_to_sequences([english_sentence])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_LENGTH, padding='post')
    predicted_sequence = english_to_hindi_model.predict(padded_sequence)
    decoded_sentence = decode_sequence(np.argmax(predicted_sequence, axis=-1)[0], hindi_tokenizer)
    return decoded_sentence

# Translate English to French
def translate_english_to_french(english_sentence, english_tokenizer, english_to_french_model):
    sequence = english_tokenizer.texts_to_sequences([english_sentence])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_LENGTH, padding='post')
    predicted_sequence = english_to_french_model.predict(padded_sequence)
    decoded_sentence = decode_sequence(np.argmax(predicted_sequence, axis=-1)[0], french_tokenizer)
    return decoded_sentence

# Translate English to Hindi and French
def translate_english_to_french_and_hindi(english_sentence, english_tokenizer, english_to_french_model, english_to_hindi_model):
    if len(english_sentence) != 10:
        raise ValueError("The input must be exactly 10 letters long.")
    
    french_translation = translate_english_to_french(english_sentence, english_tokenizer, english_to_french_model)
    hindi_translation = translate_english_to_hindi(english_sentence, english_tokenizer, english_to_hindi_model)
    return french_translation, hindi_translation

# Translate French to English
def translate_french_to_english(french_sentence, french_tokenizer, english_to_french_model):
    sequence = french_tokenizer.texts_to_sequences([french_sentence])
    padded_sequence = pad_sequences(sequence, maxlen=5, padding='post')  # Assuming 5 for French
    predicted_sequence = english_to_french_model.predict(padded_sequence)
    decoded_sentence = decode_sequence(np.argmax(predicted_sequence, axis=-1)[0], english_tokenizer)
    return decoded_sentence

# Translate French to Tamil
def translate_french_to_tamil(french_sentence, french_tokenizer, english_to_french_model, english_to_tamil_model):
    if len(french_sentence) != 5:
        raise ValueError("The French word must have exactly 5 letters.")

    english_sentence = translate_french_to_english(french_sentence, french_tokenizer, english_to_french_model)
    tamil_translation = translate_english_to_tamil(english_sentence, english_tokenizer, english_to_tamil_model)
    return tamil_translation

# Translate English to Tamil
def translate_english_to_tamil(english_sentence, english_tokenizer, english_to_tamil_model):
    sequence = english_tokenizer.texts_to_sequences([english_sentence])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_LENGTH, padding='post')
    predicted_sequence = english_to_tamil_model.predict(padded_sequence)
    decoded_sentence = decode_sequence(np.argmax(predicted_sequence, axis=-1)[0], tamil_tokenizer)
    return decoded_sentence

# Check for the word's availability and handle errors
def check_word_availability(word):
    if word not in english_tokenizer.word_index:
        wrong_words.append(word)
        raise ValueError(f"'{word}' is not available. Suggestions: {suggest_similar_words(word)}")
    return True

def suggest_similar_words(word):
    # This can be expanded with an actual word suggestion logic
    return [word + "1", word + "2"]  # Placeholder for suggestions

# Load Models and Tokenizers
def load_models_and_tokenizers():
    english_to_hindi_model = load_model('english_to_hindi_model.h5')
    english_to_french_model = load_model('english_to_french_model.h5')
    english_to_tamil_model = load_model('english_to_tamil_model.h5')

    english_tokenizer = load_tokenizer('english_tokenizer.json')
    hindi_tokenizer = load_tokenizer('hindi_tokenizer.json')
    french_tokenizer = load_tokenizer('french_tokenizer.json')
    tamil_tokenizer = load_tokenizer('tamil_tokenizer.json')

    return english_to_hindi_model, english_to_french_model, english_to_tamil_model, english_tokenizer, hindi_tokenizer, french_tokenizer, tamil_tokenizer

# Build and Train Models (if needed)
def build_and_train_models():
    english_data = load_dataset('english_converted.csv')
    hindi_data = load_dataset('hindi_converted.csv')
    french_data = load_dataset('french_converted.csv')
    tamil_data = load_dataset('tamil_converted.csv')

    # Tokenizing and padding
    english_tokenizer = Tokenizer()
    english_tokenizer.fit_on_texts(english_data['text'])  # Assuming 'text' is the column with sentences
    english_sequences = english_tokenizer.texts_to_sequences(english_data['text'])
    english_padded = pad_sequences(english_sequences, maxlen=MAX_LENGTH, padding='post')

    # Hindi Translation
    hindi_tokenizer = Tokenizer()
    hindi_tokenizer.fit_on_texts(hindi_data['text'])
    hindi_sequences = hindi_tokenizer.texts_to_sequences(hindi_data['text'])
    hindi_padded = pad_sequences(hindi_sequences, maxlen=MAX_LENGTH, padding='post')

    # Building Hindi model
    english_to_hindi_model = build_model(len(english_tokenizer.word_index) + 1, len(hindi_tokenizer.word_index) + 1)
    english_to_hindi_model.fit(english_padded, np.expand_dims(hindi_padded, -1), batch_size=64, epochs=100)

    # Save models
    english_to_hindi_model.save('english_to_hindi_model.h5')

    # Repeat the process for French and Tamil models...
    # ...
    # Similarly build and train models for English to French and English to Tamil

# Uncomment this line to build and train models if they are not yet trained
# build_and_train_models()

# Handle translations and errors
def handle_translate():
    selected_language = language_var.get()
    input_sentence = text_input.get("1.0", "end-1c").strip()
    
    try:
        if selected_language == "Hindi":
            translation = translate_english_to_hindi(input_sentence, english_tokenizer, english_to_hindi_model)
        elif selected_language == "French":
            translation = translate_english_to_french(input_sentence, english_tokenizer, english_to_french_model)
        elif selected_language == "Tamil":
            translation = translate_french_to_tamil(input_sentence, french_tokenizer, english_to_french_model, english_to_tamil_model)
        elif selected_language == "French and Hindi":
            french_translation, hindi_translation = translate_english_to_french_and_hindi(input_sentence, english_tokenizer, english_to_french_model, english_to_hindi_model)
            translation = f"French: {french_translation}, Hindi: {hindi_translation}"

    except ValueError as e:
        messagebox.showerror("Error", str(e))
        if len(wrong_words) >= 2:
            messagebox.showwarning("Wrong Words", f"Wrong words entered: {', '.join(wrong_words)}")
        translation = "Translation failed."

    translation_output.delete("1.0", "end")
    translation_output.insert("end", f"{selected_language} translation: {translation}")

# Setting up the main window
root = tk.Tk()
root.title("Multi-Language Translator")
root.geometry("650x600")

# Frame for input
input_frame = tk.Frame(root)
input_frame.pack(pady=10)

# Heading for input
input_heading = tk.Label(input_frame, text="Enter the text to be translated", font=("Times New Roman", 14, 'bold'))
input_heading.pack()
text_input = tk.Text(input_frame, height=5, width=50, font=("Times New Roman", 14))
text_input.pack()

# Language selection
language_var = tk.StringVar()
language_label = tk.Label(root, text="Select the language to translate to", font=("Times New Roman", 14, 'bold'))
language_label.pack()
language_select = ttk.Combobox(root, textvariable=language_var, values=["Hindi", "French", "Tamil", "French and Hindi"], font=("Times New Roman", 14), state="readonly")
language_select.pack()

# Submit button
submit_button = ttk.Button(root, text="Translate", command=handle_translate)
submit_button.pack(pady=10)

# Frame for output
output_frame = tk.Frame(root)
output_frame.pack(pady=10)
output_heading = tk.Label(output_frame, text="Translation: ", font=("Times New Roman", 14, 'bold'))
output_heading.pack()
translation_output = tk.Text(output_frame, height=10, width=50, font=("Times New Roman", 14))
translation_output.pack()

# Load the models and tokenizers
english_to_hindi_model, english_to_french_model, english_to_tamil_model, english_tokenizer, hindi_tokenizer, french_tokenizer, tamil_tokenizer = load_models_and_tokenizers()

# Run the Tkinter event loop
root.mainloop()
