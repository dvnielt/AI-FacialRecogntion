#!/usr/bin/env python
# coding: utf-8

# # Facial Recognition with Siamese Networks - Fixed Version

# ## 1. Setup and Dependencies

# ### 1.1 Install Dependencies

# In[1]:
# Install required packages
get_ipython().system('pip install tensorflow==2.14.1 opencv-python matplotlib scikit-learn tqdm')

# ### 1.2 Import Dependencies

# In[2]:
import cv2
import os
import random
import uuid
import numpy as np
from matplotlib import pyplot as plt
import shutil
from datetime import datetime

# Import tensorflow dependencies
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten, Dropout, Lambda
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

# ## 2. Data Collection and Preprocessing

# ### 2.1 Set Up Paths

# In[3]:
# Define paths for our data
base_dir = os.path.join(os.getcwd(), 'data')
ANC_PATH = os.path.join(base_dir, 'anchor')
POS_PATH = os.path.join(base_dir, 'positive')
NEG_PATH = os.path.join(base_dir, 'negative')

# Create directories if they don't exist
for dir_path in [base_dir, ANC_PATH, POS_PATH, NEG_PATH]:
    os.makedirs(dir_path, exist_ok=True)
    
print(f"Directory structure set up at {base_dir}")
print(f"Anchor images will be stored in: {ANC_PATH}")
print(f"Positive images will be stored in: {POS_PATH}")
print(f"Negative images will be stored in: {NEG_PATH}")

# ### 2.2 Kaggle Dataset Integration for Negative Examples

# In[4]:
def download_lfw_dataset():
    """
    Download and prepare the LFW dataset from Kaggle as negative examples.
    """
    try:
        # First try to install kagglehub if it's not already installed
        import importlib
        if importlib.util.find_spec("kagglehub") is None:
            print("Installing kagglehub...")
            import subprocess
            subprocess.check_call(['pip', 'install', 'kagglehub'])
        
        # Import kagglehub
        import kagglehub
        
        # Download the LFW dataset
        print("Downloading LFW dataset from Kaggle...")
        dataset_path = kagglehub.dataset_download("jessicali9530/lfw-dataset")
        
        print(f"Dataset downloaded to: {dataset_path}")
        
        # Process the dataset for negative examples
        lfw_images_dir = os.path.join(dataset_path, "lfw-deepfunneled")
        
        # Make sure our negative directory exists
        os.makedirs(NEG_PATH, exist_ok=True)
        
        # Count and process images
        count = 0
        for root, dirs, files in os.walk(lfw_images_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # Get person name from folder
                    person_name = os.path.basename(root)
                    
                    # Create unique filename
                    new_filename = f"{person_name}_{file}"
                    
                    # Source and destination paths
                    src_file = os.path.join(root, file)
                    dest_file = os.path.join(NEG_PATH, new_filename)
                    
                    # Copy the file
                    shutil.copy2(src_file, dest_file)
                    count += 1
                    
                    if count % 100 == 0:
                        print(f"Processed {count} files...")
        
        print(f"Successfully copied {count} images to your negative examples folder")
        return dataset_path
    
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nYou can manually download from: https://www.kaggle.com/datasets/jessicali9530/lfw-dataset")
        return None

def setup_lfw_negative_examples():
    """
    Complete workflow function to set up LFW dataset as negative examples.
    """
    print("Setting up LFW dataset as negative examples...")
    
    # Check if we already have sufficient negative examples
    if os.path.exists(NEG_PATH):
        neg_count = len([f for f in os.listdir(NEG_PATH) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        if neg_count > 100:
            print(f"You already have {neg_count} negative examples.")
            proceed = input("Do you want to download more? (y/n): ")
            if proceed.lower() != 'y':
                return
    
    # Download the dataset
    dataset_path = download_lfw_dataset()
    
    if dataset_path:
        print("\nNegative examples setup complete!")

# ### 2.3 Collect Face Images

# In[5]:
def collect_face_data():
    """
    Collect anchor and positive images using webcam.
    """
    # Set up camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Set up counters
    anchor_count = len(os.listdir(ANC_PATH))
    positive_count = len(os.listdir(POS_PATH))
    
    # Create window
    window_name = 'Face Collection'
    cv2.namedWindow(window_name)
    
    print("Face collection window opened:")
    print("Press 'a' to capture anchor images")
    print("Press 'p' to capture positive images")
    print("Press 'q' to quit")
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                print("Failed to grab frame")
                break
                
            # Create a copy for display
            display_frame = frame.copy()
            
            # Define the region of interest
            h, w = frame.shape[:2]
            center_x, center_y = w // 2, h // 2
            face_size = 250
            
            # Calculate crop coordinates
            x1 = max(0, center_x - face_size // 2)
            y1 = max(0, center_y - face_size // 2)
            x2 = min(w, x1 + face_size)
            y2 = min(h, y1 + face_size)
            
            # Draw rectangle
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add text
            cv2.putText(display_frame, f"Anchors: {anchor_count} | Positives: {positive_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(display_frame, "Press 'a' for anchor, 'p' for positive, 'q' to quit", 
                       (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display the frame
            cv2.imshow(window_name, display_frame)
            
            # Process keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            # Crop the face region
            face_crop = frame[y1:y2, x1:x2].copy()
            
            if key == ord('a'):
                # Save anchor image
                imgname = os.path.join(ANC_PATH, f'{uuid.uuid4()}.jpg')
                cv2.imwrite(imgname, face_crop)
                anchor_count += 1
                print(f"Anchor image saved! Total: {anchor_count}")
                
            elif key == ord('p'):
                # Save positive image
                imgname = os.path.join(POS_PATH, f'{uuid.uuid4()}.jpg')
                cv2.imwrite(imgname, face_crop)
                positive_count += 1
                print(f"Positive image saved! Total: {positive_count}")
                
            elif key == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"Collection complete. Anchors: {anchor_count}, Positives: {positive_count}")

# ### 2.4 Preprocessing Functions

# In[6]:
def preprocess_image(file_path):
    """
    Preprocess an image for the model.
    """
    # Read and decode image
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    
    # Resize to 100x100
    img = tf.image.resize(img, (100, 100))
    
    # Normalize to [0, 1]
    img = tf.cast(img, tf.float32) / 255.0
    
    return img

def preprocess_twin(input_img, validation_img, label):
    """
    Preprocess both images in a pair.
    """
    return (preprocess_image(input_img), preprocess_image(validation_img), label)

# ### 2.5 Create Dataset - FIXED VERSION

# In[7]:
def create_dataset(anchor_path, positive_path, negative_path, samples_per_class=300, test_split=0.3, batch_size=16):
    """
    Create training and testing datasets - FIXED VERSION
    """
    # Get file lists
    anchor_files = tf.data.Dataset.list_files(anchor_path + '/*.jpg').take(samples_per_class)
    positive_files = tf.data.Dataset.list_files(positive_path + '/*.jpg').take(samples_per_class)
    negative_files = tf.data.Dataset.list_files(negative_path + '/*.jpg').take(samples_per_class)
    
    # Convert to lists to get actual count
    anchor_list = list(anchor_files)
    positive_list = list(positive_files)
    negative_list = list(negative_files)
    
    # Get the minimum count to ensure balanced dataset
    min_samples = min(len(anchor_list), len(positive_list), len(negative_list))
    
    print(f"Using {min_samples} samples per class")
    
    # Recreate datasets with exact counts
    anchor = tf.data.Dataset.from_tensor_slices(anchor_list[:min_samples])
    positive = tf.data.Dataset.from_tensor_slices(positive_list[:min_samples])
    negative = tf.data.Dataset.from_tensor_slices(negative_list[:min_samples])
    
    # Create positive pairs
    positives = tf.data.Dataset.zip((
        anchor,
        positive,
        tf.data.Dataset.from_tensor_slices(tf.ones(min_samples, dtype=tf.float32))
    ))
    
    # Create negative pairs
    negatives = tf.data.Dataset.zip((
        anchor,
        negative,
        tf.data.Dataset.from_tensor_slices(tf.zeros(min_samples, dtype=tf.float32))
    ))
    
    # Combine datasets
    data = positives.concatenate(negatives)
    
    # Preprocess and prepare
    data = data.map(preprocess_twin)
    data = data.cache()
    data = data.shuffle(buffer_size=1024)
    
    # Calculate split
    total_samples = min_samples * 2  # positive + negative pairs
    train_size = int(total_samples * (1 - test_split))
    
    # Split into train and test
    train_data = data.take(train_size)
    train_data = train_data.batch(batch_size)
    train_data = train_data.prefetch(tf.data.AUTOTUNE)
    
    test_data = data.skip(train_size)
    test_data = test_data.batch(batch_size)
    test_data = test_data.prefetch(tf.data.AUTOTUNE)
    
    print(f"Training samples: ~{train_size}, Test samples: ~{total_samples - train_size}")
    
    return train_data, test_data

# ## 3. Build Siamese Neural Network

# ### 3.1 Create Embedding Model

# In[8]:
def make_embedding():
    """
    Build the embedding network.
    """
    inp = Input(shape=(100, 100, 3), name='embedding_input')
    
    # Convolutional blocks
    c1 = Conv2D(64, (10, 10), activation='relu')(inp)
    m1 = MaxPooling2D(pool_size=(2, 2), padding='same')(c1)
    
    c2 = Conv2D(128, (7, 7), activation='relu')(m1)
    m2 = MaxPooling2D(pool_size=(2, 2), padding='same')(c2)
    
    c3 = Conv2D(128, (4, 4), activation='relu')(m2)
    m3 = MaxPooling2D(pool_size=(2, 2), padding='same')(c3)
    
    c4 = Conv2D(256, (4, 4), activation='relu')(m3)
    f1 = Flatten()(c4)
    
    d1 = Dense(4096, activation='sigmoid')(f1)
    
    return Model(inputs=[inp], outputs=[d1], name='embedding')

# ### 3.2 Build Siamese Model - SIMPLIFIED VERSION

# In[9]:
def make_siamese_model():
    """
    Build the Siamese network - SIMPLIFIED VERSION without Lambda layers
    """
    # Create embedding model
    embedding = make_embedding()
    
    # Define inputs
    input_image = Input(name='input_img', shape=(100, 100, 3))
    validation_image = Input(name='validation_img', shape=(100, 100, 3))
    
    # Generate embeddings
    input_embedding = embedding(input_image)
    validation_embedding = embedding(validation_image)
    
    # Calculate L1 distance using a Dense layer instead of Lambda
    # First concatenate the absolute differences
    l1_distance = tf.abs(input_embedding - validation_embedding)
    
    # Classification layer
    classifier = Dense(1, activation='sigmoid', name='output')(l1_distance)
    
    # Create model
    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')

# ## 4. Training Functions

# ### 4.1 Training Function - FIXED VERSION

# In[10]:
def train_siamese_model(train_data, test_data, epochs=50, learning_rate=1e-4):
    """
    Train the Siamese model - FIXED VERSION
    """
    # Create the model
    print("Creating Siamese model...")
    siamese_model = make_siamese_model()
    
    # Print model summary
    siamese_model.summary()
    
    # Compile model
    siamese_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', Precision(name='precision'), Recall(name='recall')]
    )
    
    # Create directories
    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, f'siamese_{timestamp}_best.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Train the model
    print("Starting training...")
    history = siamese_model.fit(
        train_data,
        validation_data=test_data,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    final_path = f'siamese_model_{timestamp}.h5'
    siamese_model.save(final_path)
    print(f"Model saved to {final_path}")
    
    return siamese_model, history

# ## 5. Evaluation Functions

# ### 5.1 Evaluate Model

# In[11]:
def evaluate_model(model, test_data):
    """
    Evaluate the model performance.
    """
    # Get predictions on test set
    print("Evaluating model...")
    
    # Initialize lists for all predictions and labels
    all_predictions = []
    all_labels = []
    
    # Iterate through test data
    for batch in test_data:
        test_input, test_val, y_true = batch
        
        # Make predictions
        y_pred = model.predict([test_input, test_val], verbose=0)
        
        all_predictions.extend(y_pred)
        all_labels.extend(y_true.numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    binary_predictions = (all_predictions > 0.5).astype(int).flatten()
    accuracy = np.mean(binary_predictions == all_labels)
    
    print(f"\nTest Results:")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Visualize some examples
    test_input, test_val, y_true = next(iter(test_data))
    y_hat = model.predict([test_input, test_val])
    
    plt.figure(figsize=(15, 8))
    
    num_examples = min(4, len(test_input))
    for i in range(num_examples):
        # Anchor image
        plt.subplot(num_examples, 3, 3*i + 1)
        plt.imshow(test_input[i])
        plt.title('Anchor')
        plt.axis('off')
        
        # Comparison image
        plt.subplot(num_examples, 3, 3*i + 2)
        plt.imshow(test_val[i])
        plt.title('Comparison')
        plt.axis('off')
        
        # Prediction
        plt.subplot(num_examples, 3, 3*i + 3)
        pred_val = y_hat[i][0]
        true_val = y_true[i].numpy()
        match = "MATCH" if pred_val > 0.5 else "NO MATCH"
        color = 'green' if (pred_val > 0.5) == true_val else 'red'
        
        plt.text(0.5, 0.5, f"{match}\nConf: {pred_val:.2f}\nTrue: {true_val}", 
                ha='center', va='center', fontsize=12, color=color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# ### 5.2 Plot Training History

# In[12]:
def plot_training_history(history):
    """
    Plot training metrics.
    """
    metrics = ['loss', 'accuracy']
    
    plt.figure(figsize=(12, 4))
    
    for i, metric in enumerate(metrics):
        plt.subplot(1, 2, i+1)
        plt.plot(history.history[metric], label=f'Training {metric}')
        if f'val_{metric}' in history.history:
            plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
        plt.title(f'{metric.capitalize()} Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# ## 6. Face Verification System

# In[13]:
def verify_face(model, verification_threshold=0.5):
    """
    Real-time face verification system.
    """
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    anchor_image = None
    
    print("\nFace Verification System")
    print("Press 'c' to capture reference face")
    print("Press 'v' to start verification")
    print("Press 'q' to quit")
    
    verification_active = False
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            display_frame = frame.copy()
            h, w = frame.shape[:2]
            
            # Draw face region
            face_size = 250
            x1 = w // 2 - face_size // 2
            y1 = h // 2 - face_size // 2
            x2 = x1 + face_size
            y2 = y1 + face_size
            
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Instructions
            cv2.putText(display_frame, "Position face in green box", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c'):
                # Capture reference
                face_crop = frame[y1:y2, x1:x2]
                anchor_image = cv2.resize(face_crop, (100, 100)) / 255.0
                print("Reference face captured!")
                verification_active = False
                
            elif key == ord('v') and anchor_image is not None:
                verification_active = True
                print("Verification active!")
                
            elif key == ord('q'):
                break
            
            # Perform verification
            if verification_active and anchor_image is not None:
                face_crop = frame[y1:y2, x1:x2]
                comparison_image = cv2.resize(face_crop, (100, 100)) / 255.0
                
                # Prepare batch
                anchor_batch = np.expand_dims(anchor_image, axis=0)
                comparison_batch = np.expand_dims(comparison_image, axis=0)
                
                # Predict
                result = model.predict([anchor_batch, comparison_batch], verbose=0)[0][0]
                
                # Display result
                if result > verification_threshold:
                    text = f"MATCH ({result*100:.1f}%)"
                    color = (0, 255, 0)
                else:
                    text = f"NO MATCH ({(1-result)*100:.1f}%)"
                    color = (0, 0, 255)
                
                cv2.putText(display_frame, text, (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            cv2.imshow('Face Verification', display_frame)
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

# ## 7. Complete Workflow Function - FIXED VERSION

# In[14]:
def run_complete_workflow():
    """
    Run the complete workflow - FIXED VERSION
    """
    print("=== Facial Recognition System Setup ===\n")
    
    try:
        # Step 1: Check data
        print("Step 1: Checking data directories...")
        
        # Count existing images
        anc_count = len([f for f in os.listdir(ANC_PATH) if f.endswith(('.jpg', '.jpeg', '.png'))])
        pos_count = len([f for f in os.listdir(POS_PATH) if f.endswith(('.jpg', '.jpeg', '.png'))])
        neg_count = len([f for f in os.listdir(NEG_PATH) if f.endswith(('.jpg', '.jpeg', '.png'))])
        
        print(f"Current image counts:")
        print(f"  Anchor images: {anc_count}")
        print(f"  Positive images: {pos_count}")
        print(f"  Negative images: {neg_count}")
        
        # Step 2: Collect data if needed
        if anc_count < 20 or pos_count < 20:
            print("\nYou need at least 20 anchor and positive images.")
            if input("Collect face images now? (y/n): ").lower() == 'y':
                collect_face_data()
        
        if neg_count < 100:
            print("\nYou need negative examples for training.")
            if input("Download LFW dataset? (y/n): ").lower() == 'y':
                setup_lfw_negative_examples()
        
        # Step 3: Create datasets
        print("\nStep 3: Creating datasets...")
        train_data, test_data = create_dataset(ANC_PATH, POS_PATH, NEG_PATH)
        
        # Step 4: Train model
        if input("\nTrain the model? (y/n): ").lower() == 'y':
            epochs = int(input("Number of epochs (default 50): ") or "50")
            
            siamese_model, history = train_siamese_model(train_data, test_data, epochs=epochs)
            
            # Step 5: Evaluate
            print("\nEvaluating model...")
            evaluate_model(siamese_model, test_data)
            plot_training_history(history)
            
            # Step 6: Test verification
            if input("\nTest face verification? (y/n): ").lower() == 'y':
                verify_face(siamese_model)
            
            return siamese_model, history
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    
    return None, None

# ## Run the complete workflow
# Uncomment the line below to run the entire process
# siamese_model, history = run_complete_workflow()