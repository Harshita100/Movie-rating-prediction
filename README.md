# Movie Rating Prediction Model

This project builds a neural network to predict movie ratings (`vote_average`) using numeric, one-hot, and embedding features from a dataset containing movies' metadata.  

## Project Overview

- Predict movie ratings based on:
  - **Numeric features:** budget, popularity, revenue  
  - **Embedding features:** overview, keywords, cast, director  
  - **One-hot encoded categorical features** from the dataset  

- Uses a **fully connected neural network** with multiple hidden layers.  

---

## Dataset

- CSV file containing movie metadata.  
- Columns include:
  - `budget`, `popularity`, `revenue`
  - `overview_emb`, `keywords__emb`, `cast1_emb`, `director_emb`
  - Other categorical columns for one-hot encoding
  - Target: `vote_average`  

---

## Model Architecture

- **Input:** concatenated numeric, one-hot, and embedding features  
- **Hidden layers:** 512 → 256 → 128 units, ReLU activation  
- **Output layer:** single neuron for regression  
- **Loss:** Mean Squared Error (MSE)  
- **Metric:** Mean Absolute Error (MAE)  
- **Optimizer:** Adam  

---

## Python Code

``python
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

# Load dataset
df = pd.read_csv('/content/recent.csv')

# Remove duplicate columns
df = df.loc[:, ~df.columns.duplicated()]

# Numeric and embedding columns
numeric_cols = ['budget', 'popularity', 'revenue']
embedding_cols = ['overview_emb', 'keywords__emb', 'cast1_emb', 'director_emb']

# Ensure embedding columns are proper lists
embedding_size = len(df['overview_emb'].dropna().iloc[0])
for col in embedding_cols:
    df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [0.0]*embedding_size)

# One-hot columns
one_hot_cols = [c for c in df.columns if c not in numeric_cols + embedding_cols + ['vote_average']]

# Convert numeric columns
X_numeric = df[numeric_cols].astype(np.float32).values

# Convert one-hot columns
X_onehot = df[one_hot_cols].astype(np.float32).values

# Convert embedding columns
X_overview = np.stack(df['overview_emb'].apply(np.array).values).astype(np.float32)
X_keywords = np.stack(df['keywords__emb'].apply(np.array).values).astype(np.float32)
X_cast = np.stack(df['cast1_emb'].apply(np.array).values).astype(np.float32)
X_director = np.stack(df['director_emb'].apply(np.array).values).astype(np.float32)

# Concatenate all features
X = np.concatenate([X_numeric, X_onehot, X_overview, X_keywords, X_cast, X_director], axis=1)
y = df['vote_average'].astype(np.float32).values

# Handle NaN values
X = np.nan_to_num(X, nan=0.0)
y = np.nan_to_num(y, nan=0.0)

# Split data (70% train, 15% validation, 15% test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Build neural network
input_dim = X.shape[1]
inputs = Input(shape=(input_dim,))
x = Dense(512, activation='relu')(inputs)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
outputs = Dense(1)(x)  # regression output

model = Model(inputs, outputs)
model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse', metrics=['mae'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32)

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test)
print("Test MAE:", mae)
