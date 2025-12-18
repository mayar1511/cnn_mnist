# MNIST Handwritten Digit Classification using CNN

This project uses a **Convolutional Neural Network (CNN)** to classify handwritten digits from the **MNIST dataset**. The model achieves high accuracy by learning spatial patterns in the images.

---

## **Dataset**
- Source: [MNIST Dataset](https://www.tensorflow.org/datasets/community_catalog/huggingface/mnist)  
- Images: 28x28 grayscale images of handwritten digits (0-9)  
- Training samples: 60,000  
- Test samples: 10,000  

---

## **Workflow**
1. **Data Preprocessing**  
   - Normalized pixel values to range `[0, 1]`  
   - Reshaped images to `(28, 28, 1)` for CNN input  
   - One-hot encoded target labels  

2. **Model Architecture**  
   - 2 Convolutional layers + MaxPooling  
   - Flatten layer  
   - 1 Dense layer with 128 neurons (ReLU activation)  
   - Output Dense layer with 10 neurons (Softmax activation)  

3. **Training**  
   - Optimizer: RMSprop  
   - Loss: Categorical Crossentropy  
   - Epochs: 10  
   - Batch size: 128  

4. **Evaluation**  
   - Test Accuracy: **99.02%**  
   - Test Loss: **0.0388**  

5. **Model Saving & Loading**  
   - Saved model in Keras native format: `models/mnist_cnn_model.keras`  
   - Loaded model tested successfully with the same accuracy  

---

---

## **Usage**
1. Clone the repository:

```
git clone https://github.com/mayar1511/cnn_mnist.git
cd cnn_mnist
```

2. Install required packages:

```
pip install -r requirements.txt
```

3. Open the notebook to explore the workflow:

```
jupyter notebook notebooks/mnist_cnn.ipynb
```

4. Load the trained model in Python:

```
from tensorflow.keras.models import load_model

model = load_model("models/mnist_cnn_model.keras")
```

5. Make predictions:

```
predictions = model.predict(X_test)
```

## Dependencies
- Python 3.10+
- TensorFlow 2.x
- NumPy
- Matplotlib

## Results
- Training Accuracy: ~99.77%
- Validation Accuracy: ~99.22%
- Test Accuracy: 99.02%
- High performance demonstrates CNN's ability to classify handwritten digits effectively.
