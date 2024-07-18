import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import pathlib
import threading
import queue
import time

# Define paths to your dataset
train_dir = pathlib.Path("bateaux/seg_train")
test_dir = pathlib.Path("bateaux/seg_test")

# Function to preprocess the images
def preprocess_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [150, 150])
    img = img / 255.0
    return img

# Function to load the dataset
def load_dataset(data_dir):
    data_dir = pathlib.Path(data_dir)
    image_paths = list(data_dir.glob('*/*.jpg'))
    image_labels = [path.parent.name for path in image_paths]
    label_to_index = {name: index for index, name in enumerate(sorted(set(image_labels)))}
    image_labels = [label_to_index[label] for label in image_labels]
    
    # Convert WindowsPath objects to strings
    image_paths = [str(path) for path in image_paths]
    
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, image_labels))
    dataset = dataset.map(lambda img, label: (preprocess_image(img), label), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=len(image_paths))
    dataset = dataset.batch(32)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

train_data = load_dataset(train_dir)
test_data = load_dataset(test_dir)

# Define the CNN model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

# Compile the model with an optimizer, loss function, and metrics
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Define early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Queue to store batches of training data
train_batch_queue = queue.Queue(maxsize=10)

# Producer function to generate training batches
def produce_batches(train_data):
    print("Producer thread started...")
    for batch in train_data:
        train_batch_queue.put(batch)
        print(f"[{time.strftime('%H:%M:%S')}] Produced batch - Queue size: {train_batch_queue.qsize()}")
    print("Producer thread ended...")

# Consumer function to train the model with batches
def consume_batches():
    print("Consumer thread started...")
    while True:
        batch = train_batch_queue.get()
        if batch is None:
            break
        model.train_on_batch(batch[0], batch[1])
        train_batch_queue.task_done()
        print(f"[{time.strftime('%H:%M:%S')}] Consumed batch - Queue size: {train_batch_queue.qsize()}")
    print("Consumer thread ended...")

# Create and start producer thread
producer_thread = threading.Thread(target=produce_batches, args=(train_data,))
producer_thread.start()

# Create multiple consumer threads
num_consumers = 4  # Number of CPU cores
consumer_threads = []
for _ in range(num_consumers):
    consumer_thread = threading.Thread(target=consume_batches)
    consumer_thread.start()
    consumer_threads.append(consumer_thread)

# Wait for the producer to finish generating batches
producer_thread.join()

# Wait for all batches to be consumed
train_batch_queue.join()

# Stop consumer threads
for _ in range(num_consumers):
    train_batch_queue.put(None)
for consumer_thread in consumer_threads:
    consumer_thread.join()

print("Training started...")
model.fit(train_data,
          validation_data=test_data,
          epochs=40,
          callbacks=[early_stopping])
print("Training ended...")

# Evaluate the model with test data
model.evaluate(test_data)

# Save the trained model
model.save('my_model.h5')