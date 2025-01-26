import os
import json
import shutil  # For copying files
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import logging
import mlflow
import dagshub

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

def load_feedback_data(feedback_file, target_size=(224, 224)):
    """Load and preprocess feedback data."""
    if not os.path.exists(feedback_file):
        logging.info(f"No feedback data found at {feedback_file}.")
        return None, None

    with open(feedback_file, "r") as file:
        try:
            incorrect_predictions = json.load(file)
            logging.info(f"Loaded {len(incorrect_predictions)} feedback entries.")
        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error: {e}")
            return None, None

    if not incorrect_predictions:
        logging.info("Feedback data is empty.")
        return None, None

    images, labels = [], []
    for idx, item in enumerate(incorrect_predictions):
        image_path = item.get("image_path")
        correct_label = item.get("correct_label")

        if not image_path or correct_label is None:
            logging.warning(f"Entry {idx} is missing 'image_path' or 'correct_label'. Skipping.")
            continue

        # Handle absolute and relative paths
        if not os.path.isabs(image_path):
            feedback_dir = os.path.dirname(feedback_file)
            image_path = os.path.join(feedback_dir, image_path)

        if os.path.exists(image_path):
            try:
                image_data = load_img(image_path, target_size=target_size)
                image_data = img_to_array(image_data) / 255.0
                images.append(image_data)
                labels.append(correct_label)
                logging.debug(f"Loaded image {image_path} with label {correct_label}.")
            except Exception as e:
                logging.warning(f"Error loading image {image_path}: {e}")
        else:
            logging.warning(f"Image {image_path} not found. Skipping.")

    if not images:
        logging.info("No valid images found in feedback data.")
        return None, None

    logging.info(f"Successfully loaded {len(images)} images from feedback data.")
    return np.array(images), np.array(labels)

def build_model(base_model_path, num_classes):
    """Build and compile the model."""
    base_model = load_model(base_model_path)
    x = base_model.layers[-4].output  # Adjust as necessary
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu', kernel_regularizer='l2')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze all layers except the last 10
    for layer in model.layers[:-10]:
        layer.trainable = False
    for layer in model.layers[-10:]:
        layer.trainable = True

    model.compile(optimizer=Adam(learning_rate=1e-5),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    logging.info("Model built and compiled successfully.")
    return model

def plot_metrics(history):
    """Plot training and validation metrics."""
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    epochs_range = range(len(acc))

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig("artifacts/retraining/training_history.png")
    plt.close()
    logging.info("Training history plot saved to artifacts/retraining/training_history.png")

def retrain_model_with_feedback(base_model_path, feedback_file, save_model_path, num_classes):
    """Retrain the model using feedback data and log metrics to MLflow."""
    # Load feedback data
    images, labels = load_feedback_data(feedback_file)
    if images is None or labels is None:
        logging.info("Skipping retraining due to insufficient feedback data.")
        return

    logging.info(f"Number of feedback samples: {len(images)}")

    # Encode labels
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    # Compute class weights to handle imbalance
    class_weights = compute_class_weight('balanced', classes=np.unique(labels_encoded), y=labels_encoded)
    class_weight_dict = dict(enumerate(class_weights))
    logging.info(f"Class weights: {class_weight_dict}")

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded
    )
    logging.info(f"Training set size: {len(X_train)}")
    logging.info(f"Validation set size: {len(X_val)}")

    # Build and compile the model
    model = build_model(base_model_path, num_classes)

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

    # Initialize DagsHub MLflow integration
    dagshub.init(repo_owner="Annmary777", repo_name="Fall-Detection", mlflow=True)

    # Start an MLflow run
    with mlflow.start_run(run_name="Feedback Retraining"):
        # Log parameters
        mlflow.log_param("base_model_path", base_model_path)
        mlflow.log_param("feedback_file", feedback_file)
        mlflow.log_param("save_model_path", save_model_path)
        mlflow.log_param("num_classes", num_classes)

        # Train the model
        logging.info("Starting model training...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=5,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            class_weight=class_weight_dict
        )
        logging.info("Model training completed.")

        # Save the model
        model.save(save_model_path)
        logging.info(f"Updated model saved to {save_model_path}.")

        # Log metrics
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]

        mlflow.log_metric("final_train_accuracy", final_train_acc)
        mlflow.log_metric("final_val_accuracy", final_val_acc)
        mlflow.log_metric("final_train_loss", final_train_loss)
        mlflow.log_metric("final_val_loss", final_val_loss)

        # Plot metrics
        plot_metrics(history)

if __name__ == "__main__":
    base_model_path = "artifacts/training/model.h5"
    feedback_file = "incorrect_predictions.json"
    save_model_path = "artifacts/retraining/updated_model.h5"
    num_classes = 2

    retrain_model_with_feedback(base_model_path, feedback_file, save_model_path, num_classes)
