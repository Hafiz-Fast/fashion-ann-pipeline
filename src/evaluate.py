import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# Load trained model
model = tf.keras.models.load_model("models/model.h5")


# Load test data
x_test = np.load("data/processed/x_test.npy")
y_test = np.load("data/processed/y_test.npy")


# Evaluate model
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")


# Make predictions
predictions = model.predict(x_test, verbose=0)
y_pred = np.argmax(predictions, axis=1)


# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

display = ConfusionMatrixDisplay(confusion_matrix=cm)
display.plot()
plt.title("Fashion-MNIST Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()


# Save metrics
metrics = {
    "test_loss": float(test_loss),
    "test_accuracy": float(test_accuracy)
}

with open("metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("Evaluation completed successfully.")