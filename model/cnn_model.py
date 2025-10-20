import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import os

# Ensure consistency with data preparation script
IMAGE_HEIGHT = 100
NUM_CHANNELS = 4 # A, C, G, T one-hot encoding

def build_cnn_model(input_shape=(IMAGE_HEIGHT, NUM_CHANNELS, 1), num_classes=2):
    """
    Builds a CNN model for missense mutation prediction.

    Args:
        input_shape (tuple): The shape of the input DNA sequence image (height, width, channels).
                             For one-hot encoded DNA, width is 4 (A,C,G,T) and channels is 1.
        num_classes (int): Number of output classes (e.g., 2 for benign/pathogenic).
    
    Returns:
        tf.keras.Model: The compiled CNN model.
    """
    inputs = Input(shape=input_shape)

    # Convolutional Block 1
    # Use padding='valid' if you want kernels to only operate where they fit fully
    # Kernel size (3, NUM_CHANNELS) means it scans across 3 bases, covering all 4 channels (A,C,G,T)
    x = Conv2D(filters=32, kernel_size=(3, NUM_CHANNELS), activation='relu', padding='valid')(inputs)
    # MaxPool only across the sequence length (height), not the nucleotide features (width)
    x = MaxPooling2D(pool_size=(2, 1))(x) 

    # Convolutional Block 2
    # Kernel size (3, 1) means it scans across 3 bases, but now only 1 "feature map" column wide
    x = Conv2D(filters=64, kernel_size=(3, 1), activation='relu', padding='valid')(x)
    x = MaxPooling2D(pool_size=(2, 1))(x)

    # Convolutional Block 3
    x = Conv2D(filters=128, kernel_size=(3, 1), activation='relu', padding='valid')(x)
    x = MaxPooling2D(pool_size=(2, 1))(x)

    # Flatten and Dense layers
    x = Flatten()(x)
    x = Dense(units=256, activation='relu')(x)
    x = Dropout(0.5)(x) # Dropout to prevent overfitting
    outputs = Dense(units=num_classes, activation='softmax')(x) # Softmax for multi-class probability

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='sparse_categorical_crossentropy', # Use sparse if labels are integers (0, 1)
                  metrics=['accuracy'])
    return model

# --- Data Loading Utility Function ---
def load_data_from_npy(data_dir: str, image_height: int = IMAGE_HEIGHT, num_channels: int = NUM_CHANNELS):
    """
    Loads one-hot encoded DNA sequence images and labels from .npy files.

    Args:
        data_dir (str): Directory containing the .npy files.
        image_height (int): Expected height of the images.
        num_channels (int): Expected number of channels (4 for A, C, G, T).

    Returns:
        tuple: (images_array, labels_array)
    """
    images = []
    labels = []
    
    # Iterate through all .npy files in the directory
    for filename in os.listdir(data_dir):
        if filename.endswith('.npy'):
            filepath = os.path.join(data_dir, filename)
            
            # Load the numpy array
            img_data = np.load(filepath)
            
            # Extract label from filename (assuming 'variant_..._label_X.npy' format)
            try:
                label = int(filename.split('_label_')[-1].replace('.npy', ''))
            except ValueError:
                print(f"Warning: Could not parse label from filename {filename}. Skipping.")
                continue
            
            # Ensure the image has the expected shape and add a channel dimension for Conv2D
            if img_data.shape == (image_height, num_channels):
                images.append(img_data.reshape(image_height, num_channels, 1)) # Add channel dimension
                labels.append(label)
            else:
                print(f"Warning: Skipping {filename} due to incorrect shape: {img_data.shape}. Expected ({image_height}, {num_channels}).")

    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)


def train_model(train_dir: str, val_dir: str, model_save_path: str = 'model/mutation_prediction_model.h5',
                batch_size: int = 32, epochs: int = 50, patience: int = 10):
    """
    Trains the CNN model using the prepared data and saves the best model.

    Args:
        train_dir (str): Directory containing training data (.npy files).
        val_dir (str): Directory containing validation data (.npy files).
        model_save_path (str): Path to save the best trained model.
        batch_size (int): Batch size for training.
        epochs (int): Maximum number of training epochs.
        patience (int): Number of epochs with no improvement after which training will be stopped.
    """
    print(f"Loading training data from {train_dir}...")
    X_train, y_train = load_data_from_npy(train_dir)
    print(f"Training data loaded: X_train shape {X_train.shape}, y_train shape {y_train.shape}")

    print(f"Loading validation data from {val_dir}...")
    X_val, y_val = load_data_from_npy(val_dir)
    print(f"Validation data loaded: X_val shape {X_val.shape}, y_val shape {y_val.shape}")

    # Ensure model directory exists
    model_dir = os.path.dirname(model_save_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model = build_cnn_model()
    model.summary()

    # Callbacks for training
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(filepath=model_save_path,
                                       monitor='val_loss',
                                       save_best_only=True,
                                       verbose=1)

    print("\nStarting model training...")
    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(X_val, y_val),
                        callbacks=[early_stopping, model_checkpoint],
                        verbose=1)

    print(f"\nModel training complete. Best model saved to {model_save_path}")
    return model, history


if __name__ == '__main__':
    # Define paths based on your project structure
    train_data_dir = 'data/images/train'
    val_data_dir = 'data/images/val'
    model_output_path = 'model/mutation_prediction_model.h5'

    # You would typically run `01_data_preparation.py` first to generate these directories
    # For a full run, ensure data is prepared before training
    if not os.path.exists(train_data_dir) or not os.path.exists(val_data_dir):
        print(f"Error: Training or validation data directories not found.")
        print(f"Please run `python scripts/01_data_preparation.py` first to generate data.")
    else:
        # Example of how to train the model
        trained_model, training_history = train_model(
            train_dir=train_data_dir,
            val_dir=val_data_dir,
            model_save_path=model_output_path,
            epochs=100, # Increased epochs for better potential training, relies on early stopping
            patience=15 # Increased patience for early stopping
        )

        # Optionally evaluate on test set after training (using the best saved model)
        # from tensorflow.keras.models import load_model
        # test_data_dir = 'data/images/test'
        # X_test, y_test = load_data_from_npy(test_data_dir)
        # best_model = load_model(model_output_path)
        # test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)
        # print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
