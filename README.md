# Facial Recognition System with Siamese Neural Networks

# 📸 Overview

A sophisticated one-shot learning facial recognition system built using Siamese Neural Networks in TensorFlow. This system can verify whether two face images belong to the same person with high accuracy, requiring minimal training examples per person.

# 🌟 Features

- **One-Shot Learning**: Recognize faces with just a few training examples
- **Real-Time Verification**: Live webcam face verification system
- **Siamese Architecture**: Twin neural networks with shared weights for robust face comparison
- **Data Augmentation**: Enhanced model generalization through image preprocessing
- **Interactive UI**: User-friendly interface for data collection and verification

# 🛠️ Technologies Used

- **TensorFlow 2.14.1** - Deep learning framework
- **OpenCV** - Computer vision and image processing
- **Python 3.9** - Core programming language
- **Keras** - High-level neural network API
- **NumPy** - Numerical computing
- **Matplotlib** - Data visualization

# 📋 Requirements

```
tensorflow==2.14.1
opencv-python
matplotlib
scikit-learn
numpy
```
# 🚀 Installation

1. Clone the repository
```bash
git clone https://github.com/dvnielt/AI-FacialRecogntion
cd facial-recognition-siamese
```

2. Create a conda environment
```bash
conda create -n facial_recognition python=3.9
conda activate facial_recognition
```
3. Install dependencies
```bash
pip install tensorflow==2.14.1 opencv-python matplotlib scikit-learn
```

# 📁 Project Structure
```
facial-recognition-siamese/
│
├── data/
│   ├── anchor/         # Reference face images
│   ├── positive/       # Same person images
│   └── negative/       # Different person images
│
├── checkpoints/        # Model checkpoints
├── logs/              # TensorBoard logs
├── facial_recognition.ipynb  # Main notebook
└── README.md
```
# 💻 Usage
## Quick Start
Run the complete workflow in Jupyter Notebook:
```python
# Run the complete facial recognition pipeline
siamese_model, history = run_complete_workflow()
```

# Step-by-Step Process

1. Data Collection
```python
# Collect face images using webcam
collect_face_data()
```
Press 'a' to capture anchor (reference) images
Press 'p' to capture positive (same person) images
Press 'q' to quit


2. Download Negative Examples
```python
# Download LFW dataset for negative examples
setup_lfw_negative_examples()
```
3. Create Datasets
```python
# Prepare training and test datasets
train_data, test_data = create_dataset(ANC_PATH, POS_PATH, NEG_PATH)
```
4. Train the Model
```python
# Train Siamese network
siamese_model, history = train_siamese_model(train_data, test_data, epochs=50)
```
5. Evaluate Performance
```python
# Evaluate model and visualize results
evaluate_model(siamese_model, test_data)
plot_training_history(history)
```
7. Real-Time Verification
```python
# Launch face verification system
verify_face(siamese_model)
```
Press 'c' to capture reference face
Press 'v' to start verification mode
Press 'q' to quit



# 🏗️ Model Architecture
The Siamese Network consists of:
1. Twin Embedding Networks
- Conv2D layers: 64 → 128 → 128 → 256 filters
- MaxPooling2D for dimension reduction
- Dense layer with 4096 units for feature extraction


2. L1 Distance Layer
- Computes absolute difference between embeddings


3. Classification Layer
- Sigmoid activation for binary classification


```
Input (100x100x3) → CNN Layers → Embedding (4096) → L1 Distance → Sigmoid → Output
```

# 📊 Performance Metrics
The model is evaluated using:
- Accuracy: Overall correctness of predictions
- Precision: True positive rate
- Recall: Sensitivity to positive matches
- Real-time inference: <100ms per verification

# 🔧 Customization
## Adjust Verification Threshold
```python
verify_face(siamese_model, verification_threshold=0.7)  # More strict matching
```
## Change Training Parameters
```python
train_siamese_model(train_data, test_data, 
                   epochs=100,           # More training epochs
                   learning_rate=1e-5,   # Lower learning rate
                   batch_size=32)        # Larger batch size
```
# 📈 Results Visualization
The system provides:
- Training/validation loss curves
- Accuracy metrics over epochs
- Visual comparison of face pairs with predictions
- Confidence scores for each verification
