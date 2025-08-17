import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")


(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(f"Original x_train shape: {x_train.shape}")
print(f"Original y_train shape: {y_train.shape}")
print(f"Original x_test shape: {x_test.shape}")
print(f"Original y_test shape: {y_test.shape}")

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

x_train /= 255
x_test /= 255


num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

print(f"Processed x_train shape: {x_train.shape}")
print(f"Processed y_train shape: {y_train.shape}")

def build_cnn_model(input_shape, num_classes):
    model = Sequential([

        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

input_shape = (28, 28, 1)
model = build_cnn_model(input_shape, num_classes)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

EPOCHS = 10 
BATCH_SIZE = 128 

print("\n--- Starting Model Training ---")
history = model.fit(
    x_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(x_test, y_test),
    verbose=1 
)
print("--- Model Training Finished ---")
print("\n--- Evaluating Model Performance ---")
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()


# Preddiction
print("\n--- Making Predictions on Test Data ---")
num_predictions_to_show = 15
random_indices = np.random.choice(len(x_test), num_predictions_to_show, replace=False)
sample_images = x_test[random_indices]
sample_true_labels = np.argmax(y_test[random_indices], axis=1) 
predicted_classes = np.argmax(predictions, axis=1)
plt.figure(figsize=(10, 10))
for i in range(num_predictions_to_show):
    ax = plt.subplot(3, 5, i + 1)
    plt.imshow(sample_images[i].squeeze(), cmap='gray')
    
    true_label = sample_true_labels[i]
    predicted_label = predicted_classes[i]
    
    title_color = "green" if true_label == predicted_label else "red"
    
    plt.title(f"True: {true_label}\nPred: {predicted_label}", color=title_color)
    plt.axis('off')
plt.tight_layout()
plt.show()

print("Prediction examples displayed.")