import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import multiprocessing
import queue
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Function to load and preprocess image
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255.0
    return x, img

# Function to load the trained model
def load_model_function(model_path):
    return load_model(model_path)

# Consumer function for prediction
def consumer(prediction_queue, result_queue, model_path):
    model = load_model_function(model_path)
    while True:
        img_path = prediction_queue.get()
        if img_path is None:
            break
        x, img = load_and_preprocess_image(img_path)
        prediction = model.predict(x)
        result = "bateau entré" if prediction[0][0] > 0.5 else "bateau sorti"
        result_queue.put((img, result))
        prediction_queue.task_done()

# Function to run prediction using processes
def run_prediction(image_paths, model_path):
    # Create queues for image paths and results
    prediction_queue = multiprocessing.JoinableQueue()
    result_queue = multiprocessing.Queue()

    # Create consumer processes
    processes = []
    num_consumers = multiprocessing.cpu_count()  # Use number of CPU cores
    for _ in range(num_consumers):
        p = multiprocessing.Process(target=consumer, args=(prediction_queue, result_queue, model_path))
        p.start()
        processes.append(p)

    # Add image paths to the queue
    for img_path in image_paths:
        prediction_queue.put(img_path)

    # Signal to consumers to stop
    for _ in range(num_consumers):
        prediction_queue.put(None)

    # Wait for the consumer processes to finish
    for p in processes:
        p.join()

    # Get the results
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    
    return results

# GUI Application
class ImagePredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Boat Entry/Exit Prediction")
        self.model_path = 'my_model.h5'  # Update with your actual model path

        self.label = tk.Label(root, text="Select an image to predict")
        self.label.pack()

        self.canvas = tk.Canvas(root, width=300, height=300)
        self.canvas.pack()

        self.button = tk.Button(root, text="Browse", command=self.browse_image)
        self.button.pack()

        self.result_label = tk.Label(root, text="")
        self.result_label.pack()

    def browse_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            self.predict_image(file_path)

    def predict_image(self, img_path):
        logging.debug(f"Predicting image: {img_path}")
        results = run_prediction([img_path], self.model_path)
        if results:
            img, result = results[0]
            img = img.resize((300, 300))  # Resize the image to fit the canvas
            img = ImageTk.PhotoImage(img)
            self.canvas.create_image(150, 150, image=img, anchor=tk.CENTER)
            self.root.image = img  # Keep a reference to avoid garbage collection
            self.result_label.config(text=f"Prediction: {result}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImagePredictionApp(root)
    root.mainloop()