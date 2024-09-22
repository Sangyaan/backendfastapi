import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import numpy as np


# dir = r"C:\\Users\\TCS\\Downloads\\Potential Dysgraphia Handwriting Dataset of School-Age Children\\Potential Dysgraphia Handwriting Dataset of School-Age Children\\DATASET DYSGRAPHIA HANDWRITING\DATASET DYSGRAPHIA HANDWRITING"
# Function to preprocess the image (grayscale and resize)

def invert_image(image):
    # Step 1: Load the image
    # image = cv2.imread(image_path)

    # Step 2: Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 3: Apply a binary threshold to make text white and background black
    # The threshold value (150) can be adjusted based on the image quality
    _, binary_image = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 4: Invert the binary image (so text becomes white, background becomes black)
    inverted_image = cv2.bitwise_not(binary_image)

    # gray = cv2.cvtColor(inverted_image, cv2.COLOR_BGR2GRAY)

    resized_image = cv2.resize(inverted_image, (128, 128))
    
    # Flatten the image to create a feature vector
    flattened_image = resized_image.flatten()
    
    return flattened_image


def preprocess_image(image, size=(128, 128)):
    # Load the image
    # image = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize to a fixed size
    resized_image = cv2.resize(gray, size)
    
    # Flatten the image to create a feature vector
    flattened_image = resized_image.flatten()
    
    return flattened_image

# print(os.listdir(dir)[0])
# Example list of image paths for the dataset
# Non_dysgraphic = [os.path.join(dir, os.listdir(dir)[0], i) for i in os.listdir(os.path.join(dir, os.listdir(dir)[0])) ]  # Paths to inverted text images
# dysgraphic = [os.path.join(dir, os.listdir(dir)[1], i) for i in os.listdir(os.path.join(dir, os.listdir(dir)[1])) ]   # Paths to non-inverted text images
# print(Non_dysgraphic[0])
# Prepare dataset and labels
# X = []  # Features
# y = []  # Labels (1 for inverted, 0 for non-inverted)

# Preprocess the inverted images and append to the dataset
# for image_path in Non_dysgraphic:
    # X.append(preprocess_image(image_path))
    # y.append(0)  # Label for inverted images

# Preprocess the non-inverted images and append to the dataset
# for image_path in dysgraphic:
    # X.append(preprocess_image(image_path))
    #y.append(1)  # Label for non-inverted images

# Convert lists to numpy arrays
# X = np.array(X)
# y = np.array(y)

# Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 1: Train the SVM classifier
# classifier = svm.SVC(kernel='rbf', C=1.0)  # You can experiment with different kernels like 'rbf'
# classifier.fit(X_train, y_train)

# Step 2: Make predictions on the test set
# y_pred = classifier.predict(X_test)

# Step 3: Evaluate the model accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy * 100:.2f}%")

# Step 4: Test on a new image
# def classify_image(image):
    # processed_image = preprocess_image(image_path)
    # prediction = classifier.predict([image])
    # if prediction == 1:
        # print(f"The image is classified as Dysgraphic.")
    # else:
        # print(f"The image is classified as Non-Dysgraphic.")

# Example classification on a new image
# test_image_path = r'C:\Users\TCS\Desktop\hackademia\AI\inverted_image1.jpg'
# img = invert_image(test_image_path)
# img = np.array(img)
# img = preprocess_image(test_image_path)
# classify_image(img)