# 🧠 Handwritten Digit Classification using CNN

A deep learning project that recognizes handwritten digits (0–9) using a Convolutional Neural Network (CNN). The project includes a Streamlit web interface for interactive predictions.

## 🚀 Project Highlights

- Achieved **98.76% test accuracy** using a CNN.
- Trained and evaluated models on the MNIST handwritten digit dataset.
- Implemented image normalization, reshaping, and one-hot encoding.
- Compared Perceptron, Artificial Neural Network (ANN), and CNN performance.
- Built a Streamlit interface for handwritten digit predictions.
- Saved the trained CNN model to make predictions without retraining.

## 🛠️ Technologies Used

- Python
- TensorFlow and Keras
- CNN and Artificial Neural Networks
- NumPy and Pandas
- Matplotlib and Seaborn
- Scikit-learn
- Streamlit

## 📊 Dataset

The project uses the MNIST handwritten digit dataset, containing 60,000 training images and 10,000 test images. Each image is represented as a 28 × 28 grayscale image.

The training notebook currently loads the dataset from two CSV files:

- `mnist_train.csv`
- `mnist_test.csv`

**Dataset source:** [MNIST Dataset on Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)

Download the CSV files and place them in the same directory as the notebook before running the training cells.

## 🧠 CNN Architecture

The CNN model consists of:

1. Input layer for 28 × 28 grayscale images
2. Convolutional layer with 32 filters
3. Max-pooling layer
4. Convolutional layer with 64 filters
5. Max-pooling layer
6. Flatten layer
7. Dense layer with 128 neurons and ReLU activation
8. Dropout layer with a rate of 0.5
9. Output layer with 10 neurons and softmax activation

The model was trained for 5 epochs using the Adam optimizer and categorical cross-entropy loss.

## 📈 Model Performance

| Model | Test Accuracy |
|---|---:|
| Perceptron | 88.08% |
| ANN | 95.97% |
| CNN | **98.76%** |

The CNN achieved the highest test accuracy among the three implemented models.

## 📁 Project Structure

```text
handwritten-digit-classification-cnn/
├── app.py
├── mnist_cnn_model.h5
├── MNIST_CNN_Training.ipynb
└── README.md
```

## 💻 Run the Streamlit Application

### 1. Clone the repository

```bash
git clone https://github.com/Ayushisriva11/handwritten-digit-classification-cnn.git
cd handwritten-digit-classification-cnn
```

### 2. Install the required packages

```bash
pip install tensorflow streamlit numpy pillow
```

Install any additional packages imported by `app.py` if needed.

### 3. Launch the application

```bash
streamlit run app.py
```

The application should open in your browser. The saved model `mnist_cnn_model.h5` is used for predictions, so the model does not need to be trained again each time the app starts.

## 🔮 Future Improvements

- Deploy the application online.
- Add more evaluation metrics and prediction visualizations.
- Improve the interface and display prediction confidence.

## 👩‍💻 Author

**Ayushi Srivastava**
