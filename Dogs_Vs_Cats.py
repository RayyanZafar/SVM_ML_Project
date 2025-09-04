import os
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Set your dataset directories here (folders with images)
DOGS_DIR = r"C:\Users\Rayyan Zafar\Downloads\dog vs cat\dogs_vs_cats\test\dogs"
CATS_DIR = r"C:\Users\Rayyan Zafar\Downloads\dog vs cat\dogs_vs_cats\test\cats"

IMG_SIZE = 128

# Load MobileNetV2 model for feature extraction (no top layer)
base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(IMG_SIZE, IMG_SIZE, 3))

def extract_features(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    features = base_model.predict(img)
    return features.flatten()

features = []
labels = []

# Helper function to process folder
def process_folder(folder_path, label):
    for img_name in tqdm(os.listdir(folder_path), desc=f"Processing {label} images"):
        img_path = os.path.join(folder_path, img_name)
        try:
            feat = extract_features(img_path)
            features.append(feat)
            labels.append(label)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

# Process both dog and cat images
process_folder(DOGS_DIR, 1)  # Label 1 for dog
process_folder(CATS_DIR, 0)  # Label 0 for cat

# Convert to numpy arrays
X = np.array(features)
y = np.array(labels)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to balance the training set
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train SVM classifier
svm = SVC(kernel='linear', probability=True, class_weight='balanced', random_state=42)
svm.fit(X_train_resampled, y_train_resampled)

# Predict on test set
y_pred = svm.predict(X_test)

# Print accuracy and classification report
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Optional: Visualize feature distributions (using first 5 features for example)
feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)
df['label'] = y

def plot_kde_distributions(df, features, label_col='label'):
    sns.set(style="whitegrid")
    for feature in features:
        plt.figure(figsize=(7, 4))
        sns.kdeplot(data=df[df[label_col] == 0], x=feature, fill=True, label='Class 0 (Cat)', color='blue', alpha=0.5)
        sns.kdeplot(data=df[df[label_col] == 1], x=feature, fill=True, label='Class 1 (Dog)', color='red', alpha=0.5)
        plt.title(f'Distribution of {feature}', fontsize=14)
        plt.xlabel(feature)
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        plt.show()

# Plot first 5 features distributions
plot_kde_distributions(df, feature_names[:5])
