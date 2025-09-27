
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import librosa
import librosa.display
import tempfile
import os
import torch
# from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import torchaudio
import soundfile as sf
from pathlib import Path

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'processor' not in st.session_state:
    st.session_state.processor = None

# Create directory for saved models
MODEL_DIR = "saved_models"
Path(MODEL_DIR).mkdir(exist_ok=True)

def save_model(model, model_name):
    """Save model to the models directory"""
    path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
    joblib.dump(model, path)
    
def load_model(model_name):
    """Load model from the models directory"""
    path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return None

def get_available_models():
    """Get list of available trained models"""
    return [f.replace(".pkl", "") for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]

def preprocess_data(data):
    numerical_columns = data.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    joblib.dump(scaler, 'scaler.pkl')
    X = data.drop("LABEL", axis=1)
    y = LabelEncoder().fit_transform(data["LABEL"])
    return train_test_split(X, y, test_size=0.2)

def build_nn_model(input_shape, dropout_rate):
    model = Sequential([
        Dense(512, activation='relu', input_shape=(input_shape,)),
        Dropout(dropout_rate),
        Dense(256, activation='relu'),
        Dropout(dropout_rate),
        Dense(128, activation='relu'),
        Dropout(dropout_rate),
        Dense(64, activation='relu'),
        Dropout(dropout_rate),
        Dense(32, activation='relu'),
        Dropout(dropout_rate),
        Dense(16, activation='relu'),
        Dropout(dropout_rate),
        Dense(8, activation='relu'),
        Dropout(dropout_rate),
        Dense(4, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))
    return model, history

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred = np.where(y_pred > 0.5, 1, 0)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "roc_curve": roc_curve(y_test, y_pred)
    }

def plot_metrics(history):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(history.history['accuracy'], label='Train Accuracy')
    ax[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax[0].legend()
    ax[0].set_title('Accuracy over Epochs')
    
    ax[1].plot(history.history['loss'], label='Train Loss')
    ax[1].plot(history.history['val_loss'], label='Validation Loss')
    ax[1].legend()
    ax[1].set_title('Loss over Epochs')
    
    st.pyplot(fig)

def plot_confusion_matrix(cm):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    st.pyplot(fig)

def display_metrics(metrics):
    st.subheader("Evaluation Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{metrics['accuracy']:.2f}")
    col2.metric("Precision", f"{metrics['precision']:.2f}")
    col3.metric("Recall", f"{metrics['recall']:.2f}")
    col4.metric("F1 Score", f"{metrics['f1']:.2f}")

def extract_audio_features(audio_file, sample_length=1.0):
    y, sr = librosa.load(audio_file, sr=None)
    segment_samples = int(sample_length * sr)
    features = []

    for i in range(0, len(y), segment_samples):
        segment = y[i:i + segment_samples]
        if len(segment) < segment_samples:
            continue

        chroma_stft = np.mean(librosa.feature.chroma_stft(y=segment, sr=sr))
        rms = np.mean(librosa.feature.rms(y=segment))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=segment, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=segment, sr=sr))
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=segment, sr=sr))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=segment))
        mfccs = np.mean(librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=20), axis=1)

        segment_features = np.array([
            chroma_stft, rms, spectral_centroid, spectral_bandwidth, rolloff,
            zero_crossing_rate, *mfccs
        ])
        
        features.append(segment_features)

    return np.array(features)

def predict_audio_segments(model, audio_features):
    predictions = model.predict(audio_features)
    return np.where(predictions > 0.5, 1, 0)

def load_wav2vec_model():
    model_name = "facebook/wav2vec2-base-960h"
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name, num_labels=2)
    return model, processor

def preprocess_audio_wav2vec(audio, processor):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        sf.write(tmp_file.name, audio, samplerate=16000)
        tmp_file_path = tmp_file.name
        
    waveform, sample_rate = torchaudio.load(tmp_file_path)
    os.remove(tmp_file_path)
    
    inputs = processor(waveform.squeeze().numpy(), sampling_rate=sample_rate, return_tensors="pt", padding=True)
    return inputs

def predict_with_wav2vec(model, inputs):
    with torch.no_grad():
        logits = model(**inputs).logits
    predictions = torch.argmax(logits, dim=-1)
    return predictions.numpy()

def main():
    st.title("Audio Classification: Real vs Fake")
    
    # Initialize session state variables
    if 'model_choice' not in st.session_state:
        st.session_state.model_choice = None
    if 'history' not in st.session_state:
        st.session_state.history = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'processor' not in st.session_state:
        st.session_state.processor = None
    
    # User mode selection
    user_mode = st.sidebar.radio("Select User Mode", ["Normal User", "Super User"])
    
    if user_mode == "Super User":
        st.sidebar.success("Super User Mode: You can train and test models")
        uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
        
        if uploaded_file:
            data = pd.read_csv(uploaded_file)
            X_train, X_test, y_train, y_test = preprocess_data(data)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
            
            st.session_state.model_choice = st.selectbox("Choose Model", 
                ["Neural Network", "Random Forest", "SVM", "XGBoost", "Wav2Vec 2.0"])
            
            # Dynamic hyperparameters
            if st.session_state.model_choice == "Neural Network":
                epochs = st.number_input("Number of Epochs", min_value=1, max_value=100, value=10)
                batch_size = st.number_input("Batch Size", min_value=1, max_value=128, value=32)
                dropout_rate = st.slider("Dropout Rate", min_value=0.0, max_value=0.5, value=0.2)
            elif st.session_state.model_choice == "Random Forest":
                n_estimators = st.number_input("Number of Estimators", min_value=10, max_value=500, value=100)
                max_depth = st.number_input("Max Depth", min_value=1, max_value=50, value=10)
            elif st.session_state.model_choice == "SVM":
                kernel = st.selectbox("Kernel", ["linear", "rbf", "poly"])
                C = st.number_input("Regularization (C)", min_value=0.01, max_value=10.0, value=1.0)
            elif st.session_state.model_choice == "XGBoost":
                n_estimators = st.number_input("Number of Estimators", min_value=10, max_value=500, value=100)
                learning_rate = st.number_input("Learning Rate", min_value=0.01, max_value=1.0, value=0.1)
                max_depth = st.number_input("Max Depth", min_value=1, max_value=50, value=3)
            
            if st.button("Train Model"):
                if st.session_state.model_choice == "Neural Network":
                    model = build_nn_model(X_train.shape[1], dropout_rate)
                    model, history = train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size)
                    st.session_state.history = history
                elif st.session_state.model_choice == "Random Forest":
                    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
                    model.fit(X_train, y_train)
                elif st.session_state.model_choice == "SVM":
                    model = SVC(kernel=kernel, C=C, probability=True)
                    model.fit(X_train, y_train)
                elif st.session_state.model_choice == "XGBoost":
                    model = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
                    model.fit(X_train, y_train)
                elif st.session_state.model_choice == "Wav2Vec 2.0":
                    model, processor = load_wav2vec_model()
                    st.session_state.processor = processor
                
                st.session_state.model = model
                if st.session_state.model_choice != "Wav2Vec 2.0":
                    save_model(model, st.session_state.model_choice)
                    st.success(f"Model trained and saved as '{st.session_state.model_choice}'!")
                else:
                    st.success("Wav2Vec 2.0 model loaded!")
            
            if st.session_state.history is not None and isinstance(st.session_state.model, Sequential):
                plot_metrics(st.session_state.history)
            
            if st.button("Evaluate Model"):
                if st.session_state.model is not None:
                    if st.session_state.model_choice == "Wav2Vec 2.0":
                        st.warning("Wav2Vec 2.0 evaluation is not implemented in this example.")
                    else:
                        metrics = evaluate_model(st.session_state.model, X_test, y_test)
                        display_metrics(metrics)
                        plot_confusion_matrix(metrics['confusion_matrix'])
                else:
                    st.warning("Please train the model first.")
    
    else:  # Normal User mode
        st.sidebar.info("Normal User Mode: You can test with pre-trained models")
        available_models = get_available_models()
        
        if not available_models:
            st.warning("No trained models available. Please ask a Super User to train models first.")
        else:
            st.session_state.model_choice = st.selectbox("Select Pre-trained Model", available_models)
            model = load_model(st.session_state.model_choice)
            
            if model:
                st.session_state.model = model
                st.success(f"Loaded pre-trained {st.session_state.model_choice} model!")
            else:
                st.error(f"Failed to load {st.session_state.model_choice} model")
    
    # Common testing interface for both user modes
    if st.session_state.get('model') and st.session_state.get('model_choice'):
        st.subheader("Test Audio File")
        audio_file = st.file_uploader("Upload Audio File for Testing", type=["wav", "mp3"])
        
        if audio_file:
            st.audio(audio_file, format='audio/wav')
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_file.read())
                tmp_file_path = tmp_file.name
            
            try:
                if st.session_state.model_choice == "Wav2Vec 2.0":
                    if st.session_state.processor is None:
                        st.error("Wav2Vec processor not initialized")
                    else:
                        audio, sr = librosa.load(tmp_file_path, sr=16000) 
                        audio = audio.astype("float32")
                        inputs = preprocess_audio_wav2vec(audio, st.session_state.processor)
                        predictions = predict_with_wav2vec(st.session_state.model, inputs)
                        st.subheader("Audio Classification Results")
                        st.write(f"Predicted: {'Real' if predictions[0] == 1 else 'Fake'}")
                else:
                    y, sr = librosa.load(tmp_file_path, sr=None)
                    audio_features = extract_audio_features(tmp_file_path)
                    scaler = joblib.load('scaler.pkl')
                    audio_features_scaled = scaler.transform(audio_features)
                    predictions = predict_audio_segments(st.session_state.model, audio_features_scaled)
                    
                    st.subheader("Audio Segmentation Results")
                    for i, pred in enumerate(predictions):
                        st.write(f"Segment {i + 1}: {'Real' if pred == 1 else 'Fake'}")
                    
                    plt.figure(figsize=(10, 4))
                    librosa.display.waveshow(y, sr=sr)
                    for i, pred in enumerate(predictions):
                        start_time = i * 4.0
                        plt.axvline(x=start_time, color='r' if pred == 0 else 'g', linestyle='--')
                    plt.title("Audio Waveform with Real/Fake Segments")
                    st.pyplot(plt)
            except Exception as e:
                st.error(f"Error processing audio file: {str(e)}")
            finally:
                os.remove(tmp_file_path)
if __name__ == "__main__":
    main()