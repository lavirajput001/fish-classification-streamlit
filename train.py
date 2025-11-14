import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from model import build_patternnet

# Load your features & labels
X = np.load("features.npy")
y = np.load("labels.npy")

# Encode labels
label_encoder = LabelEncoder()
y_enc = label_encoder.fit_transform(y)
y_onehot = to_categorical(y_enc)

# Save label encoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# Build model
model = build_patternnet(X.shape[1], len(np.unique(y)))

# Train model
model.fit(X, y_onehot, epochs=20, batch_size=32, validation_split=0.2)

# Save trained model
model.save("patternnet_model.h5")
