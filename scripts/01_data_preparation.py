import pandas as pd
from Bio import SeqIO
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Define a fixed image size for the CNN
# This might need adjustment based on the typical length of sequences around your mutations
IMAGE_HEIGHT = 100  # Number of bases (e.g., 50bp upstream + REF/ALT + 50bp downstream)
NUM_CHANNELS = 4  # A, C, G, T one-hot encoding

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

def create_sequence_images(fasta_file: str, variants_df: pd.DataFrame, output_image_dir: str,
                           image_height: int = IMAGE_HEIGHT, context_length: int = 50):
    """
    Generates 2D DNA sequence "images" around missense mutations and saves them as .npy files.

    Args:
        fasta_file (str): Path to the FASTA file (e.g., chr21.fa).
        variants_df (pd.DataFrame): DataFrame with 'CHROM', 'POS', 'REF', 'ALT', 'LABEL' columns.
                                    'LABEL' should be 0 for benign, 1 for pathogenic.
        output_image_dir (str): Directory to save the generated image NumPy arrays.
        image_height (int): Desired height of the one-hot encoded sequence image.
                            This defines the total number of bases around the mutation.
        context_length (int): Number of bases to extract symmetrically around the mutation point.
                              Total extracted length will be context_length + len(REF/ALT) + context_length.
    """

    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)

    print(f"Loading FASTA sequences from {fasta_file}...")
    fasta_sequences = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        # Store with 'chr' prefix to match common variant file formats
        fasta_sequences[record.id.replace('chr', '')] = str(record.seq).upper()
    print(f"Loaded {len(fasta_sequences)} chromosomes.")

    total_variants = len(variants_df)
    for index, row in variants_df.iterrows():
        chrom = str(row['CHROM']).replace('chr', '')  # Ensure consistency
        pos = int(row['POS'])  # 1-based position of the variant
        ref = row['REF'].upper()
        alt = row['ALT'].upper()
        label = row['LABEL']  # 0 or 1

        if chrom not in fasta_sequences:
            print(f"Warning: Chromosome {chrom} not found in FASTA file. Skipping variant at {chrom}:{pos}.")
            continue

        sequence_str = fasta_sequences[chrom]

        # Calculate coordinates for extracting the sequence around the mutation
        # We need to account for 0-based indexing in Python sequences
        # The mutation point (pos) is 1-based, pointing to the first base of REF

        # Start of the reference sequence (0-based)
        ref_start_0based = pos - 1
        # End of the reference sequence (0-based, exclusive)
        ref_end_0based = ref_start_0based + len(ref)

        # Determine the region to extract around the mutation
        # This will be centered around the mutation event itself (REF replaced by ALT)

        # To get context_length bases upstream of the REF allele
        extract_start_0based = max(0, ref_start_0based - context_length)

        # To get context_length bases downstream of the ALT allele (if it were inserted)
        # For simplicity, let's target context_length bases downstream of the *reference* end
        # This might need refinement depending on how precisely you want to model indels
        extract_end_0based = min(len(sequence_str), ref_end_0based + context_length)

        # Extract the wildtype sequence around the mutation point
        extracted_wildtype_sequence = sequence_str[extract_start_0based: extract_end_0based]

        # Construct the mutant sequence by replacing REF with ALT in the extracted context
        # This assumes the mutation happens cleanly within the context.
        # For complex indels or if REF/ALT are longer than context, this needs more logic.

        # Position of REF within the *extracted_wildtype_sequence*
        position_in_extracted = ref_start_0based - extract_start_0based

        if position_in_extracted < 0 or (position_in_extracted + len(ref)) > len(extracted_wildtype_sequence):
            print(f"Warning: Mutation {chrom}:{pos} {ref}>{alt} falls outside or partially outside "
                  f"the defined context window ({context_length}bp). Skipping.")
            continue

        mutant_sequence_list = list(extracted_wildtype_sequence)
        mutant_sequence_list[position_in_extracted: position_in_extracted + len(ref)] = list(alt)
        extracted_mutant_sequence = "".join(mutant_sequence_list)

        # Decide which sequence to encode for the "image"
        # For missense, you are comparing original vs mutated context.
        # A common approach is to use the MUTANT sequence for classification,
        # and implicitly the context provides information.
        # Alternatively, you could encode both and use them as two channels or two inputs.
        # For this example, let's use the mutant sequence.

        sequence_to_encode = extracted_mutant_sequence

        # One-hot encode the sequence
        encoded_sequence = one_hot_encode_dna(sequence_to_encode)

        # Pad or crop to the desired IMAGE_HEIGHT
        if encoded_sequence.shape[0] < image_height:
            # Pad with zeros at the end
            padding_needed = image_height - encoded_sequence.shape[0]
            encoded_sequence = np.pad(encoded_sequence, ((0, padding_needed), (0, 0)), 'constant', constant_values=0)
        elif encoded_sequence.shape[0] > image_height:
            # Crop from the center or symmetrically
            start_crop = (encoded_sequence.shape[0] - image_height) // 2
            encoded_sequence = encoded_sequence[start_crop: start_crop + image_height, :]

        # Ensure the final shape is (IMAGE_HEIGHT, NUM_CHANNELS)
        if encoded_sequence.shape != (image_height, NUM_CHANNELS):
            print(f"Error: Final encoded sequence shape is {encoded_sequence.shape} for variant {chrom}:{pos}, expected ({image_height}, {NUM_CHANNELS}). Skipping.")
            continue

        # Save the encoded sequence as an image (NumPy array)
        # Include the label in the filename or store separately if needed for DataLoader
        output_file_name = f'variant_{chrom}_{pos}_{ref}_{alt}_label_{label}.npy'
        output_file_path = os.path.join(output_image_dir, output_file_name)
        np.save(output_file_path, encoded_sequence)

        if (index + 1) % 100 == 0 or (index + 1) == total_variants:
            print(f"Processed {index + 1}/{total_variants} variants. Last saved: {output_file_name}")


if __name__ == "__main__":
    fasta_path = 'data/fasta/chr21.fa'  # Adjust if the FASTA is not chr21.fa
    variants_path = 'data/variants.csv'
    output_images_base_dir = 'data/images'  # Base directory for images

    # --- Create a dummy variants.csv for demonstration if it doesn't exist ---
    if not os.path.exists(variants_path):
        print(f"Creating a dummy {variants_path} for demonstration purposes.")
        dummy_variants = pd.DataFrame({
            'CHROM': ['21', '21', '21', '21'],
            'POS': [100000, 100050, 100100, 100150],
            'REF': ['C', 'A', 'T', 'G'],
            'ALT': ['T', 'G', 'C', 'A'],
            'LABEL': [0, 1, 0, 1]  # 0 for benign, 1 for pathogenic
        })
        dummy_variants.to_csv(variants_path, index=False)
        print(f"Dummy {variants_path} created. Please replace with your actual data.")
    # --------------------------------------------------------------------------

    # Load variants data
    variants_df = pd.read_csv(variants_path)

    # It's crucial that your variants.csv includes a 'LABEL' column (0 or 1)
    if 'LABEL' not in variants_df.columns:
        print("Error: 'LABEL' column (0 or 1 for benign/pathogenic) missing in variants.csv.")
        print("Please add a 'LABEL' column to your variants.csv before running the script.")
        exit()

    # Split variants into train, test, and val sets
    # Stratify by 'LABEL' to ensure balanced classes in each split
    train_val_df, test_df = train_test_split(variants_df, test_size=0.15, random_state=42,
                                             stratify=variants_df['LABEL'])
    train_df, val_df = train_test_split(train_val_df, test_size=(0.15 / 0.85), random_state=42,
                                        stratify=train_val_df['LABEL'])  # 15% of the remaining

    print(f"Train variants: {len(train_df)}")
    print(f"Validation variants: {len(val_df)}")
    print(f"Test variants: {len(test_df)}")

    # Create images for each set
    print("\nCreating training images...")
    create_sequence_images(fasta_path, train_df, os.path.join(output_images_base_dir, 'train'))
    print("\nCreating validation images...")
    create_sequence_images(fasta_path, val_df, os.path.join(output_images_base_dir, 'val'))
    print("\nCreating test images...")
    create_sequence_images(fasta_path, test_df, os.path.join(output_images_base_dir, 'test'))

    print("\nData preparation complete.")
