import numpy as np
import os
import random
import shutil # Import shutil for rmtree

# Ensure consistency with model/cnn_model.py and data_preparation.py
IMAGE_HEIGHT = 100
NUM_CHANNELS = 4 # A, C, G, T one-hot encoding

def generate_random_dna_sequence(length: int) -> str:
    """Generates a random DNA sequence of a given length."""
    bases = ['A', 'C', 'G', 'T']
    return ''.join(random.choice(bases) for _ in range(length))

def one_hot_encode_dna(sequence: str) -> np.ndarray:
    """
    Converts a DNA sequence string to a one-hot encoded NumPy array.
    'N' (unknown base) will be all zeros.
    Returns: a NumPy array of shape (sequence_length, 4)
    """
    mapping = {
        'A': np.array([1., 0., 0., 0.], dtype=np.float32),  # A: position 0 is 1
        'C': np.array([0., 1., 0., 0.], dtype=np.float32),  # C: position 1 is 1
        'G': np.array([0., 0., 1., 0.], dtype=np.float32),  # G: position 2 is 1
        'T': np.array([0., 0., 0., 1.], dtype=np.float32),  # T: position 3 is 1
        'N': np.array([0., 0., 0., 0.], dtype=np.float32)   # N: all zeros for unknown base
    }

    # Handle lowercase bases by converting to uppercase and defaulting to 'N' if not found
    encoded_sequence = [mapping.get(nucleotide.upper(), mapping['N']) for nucleotide in sequence]
    return np.array(encoded_sequence, dtype=np.float32)

def create_dummy_image_data(output_dir: str, num_samples: int, label_ratio: dict = None):
    """
    Generates and saves dummy one-hot encoded DNA sequence images as .npy files.

    Args:
        output_dir (str): Directory to save the dummy data.
        num_samples (int): Total number of dummy samples to generate.
        label_ratio (dict): A dictionary specifying the ratio of labels, e.g., {0: 0.5, 1: 0.5}
                            If None, labels will be random.
    """
    # Ensure the output directory exists. If it doesn't, create it and any missing parent directories.
    # If it exists, do nothing (because of exist_ok=True).
    os.makedirs(output_dir, exist_ok=True) # <<< FIX: Ensure directory creation/existence

    print(f"Generating {num_samples} dummy samples for {output_dir}...")

    # Generate labels based on ratio if provided
    labels = []
    if label_ratio:
        num_class_0 = int(num_samples * label_ratio.get(0, 0)) # Ensure key exists
        num_class_1 = num_samples - num_class_0
        labels.extend([0] * num_class_0)
        labels.extend([1] * num_class_1)
        random.shuffle(labels) # Shuffle to mix the labels
    else:
        labels = [random.randint(0, 1) for _ in range(num_samples)]

    for i in range(num_samples):
        # Generate a random DNA sequence
        dna_sequence = generate_random_dna_sequence(IMAGE_HEIGHT)

        # One-hot encode the sequence
        encoded_sequence = one_hot_encode_dna(dna_sequence)

        # Ensure the final shape is (IMAGE_HEIGHT, NUM_CHANNELS)
        # This padding/cropping logic should handle sequences not exactly IMAGE_HEIGHT
        if encoded_sequence.shape[0] < IMAGE_HEIGHT: # Check height dimension
            padding_needed = IMAGE_HEIGHT - encoded_sequence.shape[0]
            encoded_sequence = np.pad(encoded_sequence, ((0, padding_needed), (0, 0)), 'constant', constant_values=0)
        elif encoded_sequence.shape[0] > IMAGE_HEIGHT: # Check height dimension
            start_crop = (encoded_sequence.shape[0] - IMAGE_HEIGHT) // 2
            encoded_sequence = encoded_sequence[start_crop: start_crop + IMAGE_HEIGHT, :]
            
        # Get the label for this sample correctly
        current_label = labels[i] # FIX IS HERE

        # Save the encoded sequence as a NumPy array file
        output_file_name = f'dummy_variant_sample_{i}_label_{current_label}.npy'
        output_file_path = os.path.join(output_dir, output_file_name)
        np.save(output_file_path, encoded_sequence)

    print(f"Finished generating dummy data for {output_dir}.")


if __name__ == "__main__":
    base_images_dir = 'data/images'

    # --- REVISED CLEANUP LOGIC ---
    # Instead of deleting and recreating, ensure directories exist
    # and then just clear out .npy files within them for a fresh start.
    for subdir in ['train', 'val', 'test']:
        full_path = os.path.join(base_images_dir, subdir)
        
        # Ensure the directory structure exists first
        os.makedirs(full_path, exist_ok=True)
        print(f"Ensured directory exists: {full_path}")

        # Now, clean out only the .npy files within that directory
        try:
            for f in os.listdir(full_path):
                if f.endswith('.npy'): # Only remove .npy files
                    os.remove(os.path.join(full_path, f))
            print(f"Cleaned up existing .npy files in: {full_path}")
        except OSError as e:
            print(f"Error cleaning up .npy files in {full_path}: {e}")
            print("Skipping .npy file cleanup. Handle manually if issues persist.")
    # --- END REVISED CLEANUP LOGIC ---

    # Create dummy data for train, val, test sets
    create_dummy_image_data(os.path.join(base_images_dir, 'train'), num_samples=1000, label_ratio={0: 0.5, 1: 0.5})
    create_dummy_image_data(os.path.join(base_images_dir, 'val'), num_samples=200, label_ratio={0: 0.5, 1: 0.5})
    create_dummy_image_data(os.path.join(base_images_dir, 'test'), num_samples=200, label_ratio={0: 0.5, 1: 0.5})

    print("\nDummy data generation complete. You can now proceed to train the model.")
    print("Run: python model/cnn_model.py")
    print("Then run: python backend/app.py")