#pp.py
import os
import librosa
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import entropy
from nolds import dfa
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
import joblib

rf_model = joblib.load("random_forest_model.pkl")
mlp_model = tf.keras.models.load_model("mlp_model.keras", custom_objects={ "focal_loss_fn": lambda y_true, y_pred: tf.keras.losses.categorical_crossentropy(y_true, y_pred) })
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("labelencoder.pkl")

# Constants
SAMPLE_RATE = 16000
DURATION = 20
MFCC_FEATURES = 40
SEGMENT_LENGTH = DURATION * SAMPLE_RATE
HOP_LENGTH = 128

# Feature extraction function
def extract_all_features(audio_path):
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION)
        if len(y) < SEGMENT_LENGTH:
            y = np.pad(y, (0, SEGMENT_LENGTH - len(y)), mode='constant')

        # MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=MFCC_FEATURES, hop_length=HOP_LENGTH)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_var = np.var(mfccs, axis=1)
        delta_mfccs = np.mean(librosa.feature.delta(mfccs), axis=1)

        # Spectral features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=HOP_LENGTH))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y, hop_length=HOP_LENGTH))

        # Envelope and STFT
        envelope = np.abs(librosa.util.frame(y, frame_length=256, hop_length=HOP_LENGTH).mean(axis=0))
        envelope_smooth = librosa.util.normalize(np.convolve(envelope, np.ones(500)/500, mode='same'))
        stft = np.abs(librosa.stft(y, n_fft=256, hop_length=HOP_LENGTH))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=256)

        # Lung sound features
        peaks, _ = find_peaks(envelope, height=np.mean(envelope) * 2, distance=50)
        crackle_count = len(peaks)
        fine_crackle_ratio = 0
        if crackle_count > 0 and len(peaks[peaks < stft.shape[1]]) > 0:
            valid_peaks = peaks[peaks < stft.shape[1]]
            crackle_freqs = np.mean(stft[:, valid_peaks], axis=1)
            fine_crackle_ratio = np.sum(crackle_freqs[freqs > 400]) / np.sum(crackle_freqs) if np.sum(crackle_freqs) > 0 else 0

        high_freq_energy = np.mean(stft[(freqs > 400) & (freqs < 1000), :], axis=0)
        wheeze_ratio = np.sum(high_freq_energy > np.mean(high_freq_energy) * 1.5) / high_freq_energy.size

        breath_peaks, _ = find_peaks(envelope_smooth, distance=sr//2, height=np.mean(envelope_smooth))
        respiratory_rate = len(breath_peaks) * (60 / DURATION)
        ie_ratio = 1.0
        if len(breath_peaks) > 1:
            cycle_durations = np.diff(breath_peaks) * (HOP_LENGTH / sr)
            insp_peaks, _ = find_peaks(-envelope_smooth, distance=sr//2)
            exp_durations = np.diff(insp_peaks) * (HOP_LENGTH / sr) if len(insp_peaks) > 1 else cycle_durations
            ie_ratio = np.mean(cycle_durations) / np.mean(exp_durations) if len(exp_durations) > 0 else 1.0

        energy_low = np.mean(np.sum(stft[(freqs >= 100) & (freqs < 400), :], axis=0)**2)
        energy_high = np.mean(np.sum(stft[(freqs >= 400) & (freqs < 1000), :], axis=0)**2)

        tonal_energy = np.mean(stft[(freqs > 50) & (freqs < 200), :], axis=0)
        ventilator_ratio = np.sum(tonal_energy > np.mean(tonal_energy) * 2) / tonal_energy.size

        # MDVP-like features
        f0, voiced_flag, _ = librosa.pyin(y, fmin=50, fmax=1000, sr=sr, hop_length=HOP_LENGTH)
        f0_mean = np.nanmean(f0) if np.any(voiced_flag) else 0
        f0_hi = np.nanmax(f0) if np.any(voiced_flag) else 0
        f0_lo = np.nanmin(f0) if np.any(voiced_flag) else 0

        jitter_abs = jitter_rap = jitter_ppq = jitter_ddp = 0
        if np.any(voiced_flag) and len(f0[~np.isnan(f0)]) > 1:
            f0_diff = np.diff(f0[~np.isnan(f0)])
            jitter_abs = np.mean(np.abs(f0_diff))
            jitter_rap = np.mean(np.abs(f0_diff)) / f0_mean if f0_mean > 0 else 0
            jitter_ppq = np.mean(np.abs(f0_diff[:3])) / f0_mean if len(f0_diff) >= 3 and f0_mean > 0 else 0
            jitter_ddp = jitter_rap * 3

        peaks, _ = find_peaks(y, distance=50)
        shimmer = np.mean(np.abs(np.diff(np.abs(y[peaks])))) / np.mean(np.abs(y[peaks])) if len(peaks) > 1 else 0

        y_short = librosa.resample(y[:SAMPLE_RATE * 5], orig_sr=SAMPLE_RATE, target_sr=8000)
        hnr = np.mean(librosa.effects.harmonic(y_short)) / np.mean(librosa.effects.percussive(y_short)) if np.mean(librosa.effects.percussive(y_short)) > 0 else 0
        nhr = 1 / hnr if hnr > 0 else 0

        # Nonlinear features
        rpde = entropy(np.histogram(np.diff(peaks), bins=50, density=True)[0]) if len(peaks) > 1 else 0
        dfa_val = dfa(y_short) if len(y_short) > 100 else 0
        ppe = entropy(np.histogram(f0[~np.isnan(f0)], bins=50, density=True)[0]) if np.any(voiced_flag) else 0

        spread1 = f0_hi - f0_lo if np.any(voiced_flag) else 0
        spread2 = np.std(f0[~np.isnan(f0)]) if np.any(voiced_flag) else 0

        # Feature vector (full set to match training)
        features = np.concatenate([
            mfccs_mean, mfccs_var, delta_mfccs,
            [spectral_centroid, zcr, spectral_rolloff, energy_low, energy_high,
             crackle_count, fine_crackle_ratio, wheeze_ratio, respiratory_rate,
             ie_ratio, ventilator_ratio,
             f0_mean, f0_hi, f0_lo, jitter_abs, jitter_rap, jitter_ppq, jitter_ddp,
             shimmer, hnr, nhr, rpde, dfa_val, ppe, spread1, spread2]
        ])
        return {
            "status": "success",
            "features": features.tolist()  # convert np.ndarray to JSON-serializable list
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }





def predict_audio(features: list):
    try:
        # Define the expected feature column names (same as during training)
        feature_columns = [
            *[f'mfcc_mean_{i}' for i in range(MFCC_FEATURES)],
            *[f'mfcc_var_{i}' for i in range(11)],  # Only mfcc_var_0 to mfcc_var_10
            *[f'delta_mfcc_{i}' for i in range(MFCC_FEATURES)],
            'spectral_centroid', 'zero_crossing_rate', 'spectral_rolloff',
            'energy_low_band', 'energy_high_band',
            'crackle_count', 'fine_crackle_ratio', 'wheeze_ratio', 'respiratory_rate',
            'ie_ratio', 'ventilator_ratio',
            'f0_mean', 'f0_hi', 'f0_lo',
            'jitter_abs', 'jitter_rap', 'jitter_ppq', 'jitter_ddp',
            'shimmer', 'hnr', 'nhr', 'rpde', 'dfa', 'ppe', 'spread1', 'spread2'
        ]

        # Create aligned feature dataframe
        feature_df = pd.DataFrame([np.zeros(len(feature_columns))], columns=feature_columns)
        for i, feat in enumerate(feature_columns):
            if i < len(features):
                feature_df.iloc[0, i] = features[i]

        # Scale full feature set
        X_scaled_full = scaler.transform(feature_df)

        # Selected feature names for MLP
        selected_features = [
            'mfcc_mean_1', 'mfcc_mean_3', 'mfcc_mean_2', 'mfcc_mean_8', 'mfcc_mean_37',
            'mfcc_mean_7', 'spectral_centroid', 'mfcc_var_10', 'energy_low_band', 'mfcc_var_9',
            'mfcc_mean_4', 'zero_crossing_rate', 'mfcc_mean_0', 'mfcc_var_11', 'spectral_rolloff',
            'mfcc_mean_32', 'wheeze_ratio', 'energy_high_band', 'mfcc_mean_6', 'ventilator_ratio',
            'crackle_count', 'respiratory_rate'
        ]
        # Extract MLP input subset
        X_scaled_subset = X_scaled_full[:, [feature_columns.index(f) for f in selected_features if f in feature_columns]]

        # RF prediction
        rf_pred = rf_model.predict(X_scaled_full)
        rf_label = label_encoder.inverse_transform(rf_pred)[0]
        rf_conf = rf_model.predict_proba(X_scaled_full)[0].max()

        # MLP prediction
        mlp_out = mlp_model.predict(X_scaled_subset, verbose=0)[0]
        mlp_pred = np.argmax(mlp_out)
        mlp_label = label_encoder.inverse_transform([mlp_pred])[0]
        mlp_conf = np.max(mlp_out)

        # Ensemble
        weights = [0.7, 0.3]
        ensemble_pred = np.bincount([rf_pred[0], mlp_pred], weights=weights).argmax()
        ensemble_label = label_encoder.inverse_transform([ensemble_pred])[0]

        return {
            "status": "success",
            "RF Prediction": rf_label,
            "RF Confidence": float(rf_conf),
            "MLP Prediction": mlp_label,
            "MLP Confidence": float(mlp_conf),
            "Ensemble Prediction": ensemble_label
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

