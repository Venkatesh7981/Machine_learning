import os
import librosa
import pandas as pd

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, duration=3, offset=0.5)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        return mfccs.mean(axis=1)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def save_features():
    emotions = ['angry', 'happy', 'sad']
    base_path = os.path.join('Audio_files', 'audio_files')
    data = []

    print("Searching in:", os.path.abspath(base_path))

    for emotion in emotions:
        folder = os.path.join(base_path, emotion)
        print(f"Scanning folder: {folder}")

        if not os.path.exists(folder):
            print(f"❌ Folder does not exist: {folder}")
            continue

        for filename in os.listdir(folder):
            if filename.endswith('.wav'):
                file_path = os.path.join(folder, filename)
                print(f"🔍 Processing: {file_path}")
                features = extract_features(file_path)
                if features is not None:
                    data.append([*features, emotion])
                else:
                    print(f"⚠️ Skipping file due to error: {file_path}")

    if data:
        df = pd.DataFrame(data)
        os.makedirs('model', exist_ok=True)
        df.to_csv('model/features.csv', index=False)
        print("✅ Features saved to model/features.csv")
    else:
        print("⚠️ No features extracted. CSV not created.")

if __name__ == "__main__":
    save_features()
