import os
from keras.models import load_model
import librosa

# Function to extract features from the file (assuming audio files)
def extract_features(file_path):
    # Load the audio file
    audio, sr = librosa.load(file_path, sr=None)
    
    # Example: Extract MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return mfccs

# Load the pre-trained model
model = load_model("model/emotion_model.h5")

# Specify the file path to the audio file
file_path = "path_to_your_audio_file.wav"  # Make sure this is the correct path

# Extract features from the file
features = extract_features(file_path)

# Prepare the features for prediction (reshape if necessary, depending on the model)
features = features.reshape(1, -1)  # Example reshape; modify according to your model's requirements

# Predict the emotion
emotion = model.predict(features)

# Print the result
print(f"Predicted Emotion: {emotion}")
