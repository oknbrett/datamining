# datamining
Here you can find the link to the meeting

Extra model 1: Neural Network
https://teams.microsoft.com/l/meetingrecap?driveId=b%21HK9x6ZtgGkKfAH0MrIgvX3SsC4NvumNDr46WtF0zQxDhsFvVd3bgTYIrT7aimXj7&driveItemId=01OGLASGRO2OVUE6MGABE2RIY3GE3FOUAN&sitePath=https%3A%2F%2Fhannl-my.sharepoint.com%2F%3Av%3A%2Fg%2Fpersonal%2Fkn_ong_student_han_nl%2FES7Tq0J5hgBJqKMbMTZXUA0BqA6RDNZrv9fu9ZNd5ZXyRw&fileUrl=https%3A%2F%2Fhannl-my.sharepoint.com%2F%3Av%3A%2Fg%2Fpersonal%2Fkn_ong_student_han_nl%2FES7Tq0J5hgBJqKMbMTZXUA0BqA6RDNZrv9fu9ZNd5ZXyRw&threadId=19%3A3e4cba00-c221-4316-84bf-5cd21e8ab0f7_75ebba36-29ce-4051-b5e0-3e3b1a79719a%40unq.gbl.spaces&callId=0c3f1ffa-2af1-47a1-8294-d8023265c12b&threadType=OneOnOneChat&meetingType=Unknown&subType=RecapSharingLink_RecapChiclet

Model 1: KNN
https://teams.microsoft.com/l/meetingrecap?driveId=b%21HK9x6ZtgGkKfAH0MrIgvX3SsC4NvumNDr46WtF0zQxDhsFvVd3bgTYIrT7aimXj7&driveItemId=01OGLASGSCAXS5FDQQONCZR6IW4UM3NUMC&sitePath=https%3A%2F%2Fhannl-my.sharepoint.com%2F%3Av%3A%2Fg%2Fpersonal%2Fkn_ong_student_han_nl%2FEUIF5dKOEHNFmPkW5Rm20YIBjozdeyEJMsz9hTntOVhKwQ&fileUrl=https%3A%2F%2Fhannl-my.sharepoint.com%2F%3Av%3A%2Fg%2Fpersonal%2Fkn_ong_student_han_nl%2FEUIF5dKOEHNFmPkW5Rm20YIBjozdeyEJMsz9hTntOVhKwQ&threadId=19%3A3e4cba00-c221-4316-84bf-5cd21e8ab0f7_75ebba36-29ce-4051-b5e0-3e3b1a79719a%40unq.gbl.spaces&callId=0c3f1ffa-2af1-47a1-8294-d8023265c12b&threadType=OneOnOneChat&meetingType=Unknown&subType=RecapSharingLink_RecapChiclet

Model 2 and Extra model 2: Naive Bayes & Logistic Regression
https://teams.microsoft.com/l/meetingrecap?driveId=b%21HK9x6ZtgGkKfAH0MrIgvX3SsC4NvumNDr46WtF0zQxDhsFvVd3bgTYIrT7aimXj7&driveItemId=01OGLASGXDHE3NFT52NNFZZSZUNNTTHLE4&sitePath=https%3A%2F%2Fhannl-my.sharepoint.com%2F%3Av%3A%2Fg%2Fpersonal%2Fkn_ong_student_han_nl%2FEeM5NtLPumtLnMs0a2czrJwB4xZsn_1bxZbpTa_fU-URlA&fileUrl=https%3A%2F%2Fhannl-my.sharepoint.com%2F%3Av%3A%2Fg%2Fpersonal%2Fkn_ong_student_han_nl%2FEeM5NtLPumtLnMs0a2czrJwB4xZsn_1bxZbpTa_fU-URlA&threadId=19%3A3e4cba00-c221-4316-84bf-5cd21e8ab0f7_75ebba36-29ce-4051-b5e0-3e3b1a79719a%40unq.gbl.spaces&callId=0c3f1ffa-2af1-47a1-8294-d8023265c12b&threadType=OneOnOneChat&meetingType=Unknown&subType=RecapSharingLink_RecapChiclet

In case you need permission to watch the clip, please send me a request and I will approve asap.

For some reason I cannot push the file due to a problem with branches.\

KNN:
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load the dataset
file_path = "/workspaces/datamining/customer_purchase_data.csv"
rawDF = pd.read_csv(file_path)

# Display first few rows
print(rawDF.head())

# Drop unnecessary columns (if any)
cleanDF = rawDF.drop(["customer_id"], axis=1, errors='ignore')

# Convert categorical variables if necessary
cleanDF["Gender"] = cleanDF["Gender"].astype("category")
cleanDF["ProductCategory"] = cleanDF["ProductCategory"].astype("category")
cleanDF["LoyaltyProgram"] = cleanDF["LoyaltyProgram"].astype("category")
cleanDF["PurchaseStatus"] = cleanDF["PurchaseStatus"].astype("category")

# Basic statistics
print(cleanDF.describe())

# Scatter matrix to visualize relationships
selDF = cleanDF[["Age", "AnnualIncome", "NumberOfPurchases", "TimeSpentOnWebsite"]]
fig = scatter_matrix(selDF, alpha=0.2, figsize=(6, 6), diagonal="hist")
for ax in fig.flatten():
    ax.xaxis.label.set_rotation(90)
    ax.yaxis.label.set_rotation(0)
    ax.yaxis.label.set_ha("right")
plt.tight_layout()
plt.gcf().subplots_adjust(wspace=0, hspace=0)
plt.show()

# Normalize numeric data
def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

numeric_cols = ["Age", "AnnualIncome", "NumberOfPurchases", "TimeSpentOnWebsite", "DiscountsAvailed"]
cleanDF[numeric_cols] = cleanDF[numeric_cols].apply(normalize, axis=0)

# Define X (features) and y (target)
excluded = ["PurchaseStatus"]  # Exclude target variable
X = cleanDF.drop(columns=excluded)
y = cleanDF["PurchaseStatus"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the KNN model
knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=knn.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn.classes_)
disp.plot()
plt.show()

# %%
new_customer = pd.DataFrame([{
    "Age": 44,
    "Gender": 0,
    "AnnualIncome": 86086.8710991164,
    "NumberOfPurchases": 16,
    "ProductCategory": 4,
    "TimeSpentOnWebsite": 28.9704995823636,
    "LoyaltyProgram": 0,
    "DiscountsAvailed": 3
}])

# %%
# Normalize numeric features using the same method as training
for col in ["Age", "AnnualIncome", "NumberOfPurchases", "TimeSpentOnWebsite", "DiscountsAvailed"]:
    new_customer[col] = (new_customer[col] - cleanDF[col].min()) / (cleanDF[col].max() - cleanDF[col].min())


# %%
prediction = knn.predict(new_customer)
print(f"Predicted Purchase Status: {prediction[0]}")  # 0 = No, 1 = Yes


# %%
from sklearn.metrics import accuracy_score

# Predict on the test set
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2%}")


# %%
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

k_values = range(1, 21)
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))

# Plot
plt.plot(k_values, accuracies, marker='o')
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("K vs Accuracy")
plt.xticks(k_values)
plt.show()

# Best K based on the test set
best_k = k_values[accuracies.index(max(accuracies))]
print(f"Best K found: {best_k}")



Neural Network For Ai vs Real Image Detector:

# %%
import kagglehub
import os
import tensorflow as tf

# Download the CIFAKE dataset
path = kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images")
print("Path to dataset files:", path)

# Define paths to training and test directories
train_dir = os.path.join(path, 'train')
test_dir = os.path.join(path, 'test')

# Load dataset using TensorFlow's image_dataset_from_directory
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(32, 32),  # CIFAKE images are 32x32
    batch_size=32,
    label_mode='binary'   # 0 for real, 1 for AI-generated
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(32, 32),
    batch_size=32,
    label_mode='binary'
)

# Verify dataset
class_names = train_dataset.class_names
print("Class names:", class_names)

# %%
# Purpose: Normalize images (0-1 range) and optimize loading speed
train_dataset = train_dataset.map(lambda x, y: (x / 255.0, y))  # Normalize pixel values
train_dataset = train_dataset.cache().prefetch(tf.data.AUTOTUNE)  # Speed up training

test_dataset = test_dataset.map(lambda x, y: (x / 255.0, y))  # Normalize test set
test_dataset = test_dataset.cache().prefetch(tf.data.AUTOTUNE)

print("Dataset preprocessed and ready!")

# %%
import tensorflow as tf

# Purpose: Define a CNN model to classify AI-generated vs real images
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary output: 0 (REAL) or 1 (FAKE)
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("Model built and compiled!")

# %%
# Purpose: Train the model on the dataset to learn AI vs real image patterns
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=5  # Start with 5 epochs, adjust later if needed
)

print("Training complete!")

# %%
# Purpose: Test the model’s accuracy on the unseen test dataset
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# %%
# Purpose: Test multiple epoch values and find the best accuracy
epoch_range = [5, 10, 15, 20]  # Epochs to try
results = {}

# Rebuild the simpler model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Loop through epochs
for epochs in epoch_range:
    print(f"Training with {epochs} epochs...")
    model.fit(train_dataset, validation_data=test_dataset, epochs=epochs, verbose=0)
    test_loss, test_accuracy = model.evaluate(test_dataset, verbose=0)
    results[epochs] = test_accuracy
    print(f"Epochs: {epochs}, Test Accuracy: {test_accuracy * 100:.2f}%")

# Find best result
best_epochs = max(results, key=results.get)
print(f"Best number of epochs: {best_epochs} with accuracy: {results[best_epochs] * 100:.2f}%")

# %%
# Purpose: Compare models with fixed randomness
tf.random.set_seed(42)  # Fix randomness

# Original model (5 epochs)
model1 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model1.fit(train_dataset, epochs=5, verbose=0)
loss1, acc1 = model1.evaluate(test_dataset, verbose=0)
print(f"Model 1 (5 epochs): {acc1 * 100:.2f}%")

# Best from grid (15 epochs)
model2 = tf.keras.Sequential([  # Same architecture
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model2.fit(train_dataset, epochs=15, verbose=0)
loss2, acc2 = model2.evaluate(test_dataset, verbose=0)
print(f"Model 2 (15 epochs): {acc2 * 100:.2f}%")

# %%


import tensorflow as tf
import numpy as np
from PIL import Image

# Rebuild Model 1 with seed for consistency
tf.random.set_seed(42)
model1 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model1.fit(train_dataset, epochs=5, verbose=0)

# Load and preprocess an image (replace 'your_image.jpg' with your file path)
img_path = '/workspaces/datamining/ChatGPT Image Apr 9, 2025, 08_50_11 PM.png'  # Upload an image to your environment
img = Image.open(img_path).resize((32, 32))  # Resize to 32x32
img_array = np.array(img) / 255.0  # Normalize
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Predict
prediction = model1.predict(img_array)
result = "AI-generated" if prediction[0][0] > 0.5 else "Real"
confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]
print(f"Prediction: {result} (Confidence: {confidence * 100:.2f}%)")

# %% [markdown]
# Below is a real image, photographed by a real person, and is predicted as AI-generated. This goes to show the limitation in the training dataset. The dataset was very out-dated and could not handle variety well. With a better dataset, things might be different. However, my PC wouldn't be able to handle such data. Given a good dataset and a good PC, I believe this could be achieved. This is what already happening with xAI and X (twitter) when Elon Musk sold X to xAI. This gives xAI a huge environment with real-time data every seconds by real people.

# %%
# Load and preprocess an image (replace 'your_image.jpg' with your file path)
img_path = '/workspaces/datamining/josh-hild-16ZUFFYQdbo-unsplash.jpg'  # Upload an image to your environment
img = Image.open(img_path).resize((32, 32))  # Resize to 32x32
img_array = np.array(img) / 255.0  # Normalize
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Predict
prediction = model1.predict(img_array)
result = "AI-generated" if prediction[0][0] > 0.5 else "Real"
confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]
print(f"Prediction: {result} (Confidence: {confidence * 100:.2f}%)")

# %% [markdown]
# Here I tried to create a website to upload an image and the model will tell you whether or not the picture is AI-generated, when I turned it into a Flask app, I kept hitting errors like ‘SystemExit: 1’ because of port conflicts in Codespace. I couldn’t get the server running smoothly in time, even with help. The model works, but the app part just wouldn’t cooperate..
# 
# 

# %%
from flask import Flask, request
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import kagglehub
import socket

app = Flask(__name__)

# Load CIFAKE dataset
path = kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images")
train_dir = os.path.join(path, 'train')
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(32, 32),
    batch_size=32,
    label_mode='binary'
).map(lambda x, y: (x / 255.0, y)).cache().prefetch(tf.data.AUTOTUNE)

# Load Model 1
tf.random.set_seed(42)
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_dataset, epochs=5, verbose=0)

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files['image']
        img = Image.open(file.stream).resize((32, 32))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_array)
        result = "AI-generated" if prediction[0][0] > 0.5 else "Real"
        confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]
        return f"Prediction: {result} (Confidence: {confidence * 100:.2f}%)"
    return '''
        <h1>AI vs Real Image Detector</h1>
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="image">
            <input type="submit" value="Upload">
        </form>
    '''

# Find a free port
def get_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port

if __name__ == '__main__':
    port = get_free_port()
    print(f"Starting server on port {port}...")
    try:
        app.run(host='0.0.0.0', port=port, debug=True)
    except Exception as e:
        print(f"Error: {e}")






