# Imports
import tensorflow 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.applications import MobileNetV2
from keras.preprocessing import image
import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
train_dir = pathlib.Path(r"bateaux\seg_train")
test_dir = pathlib.Path(r"bateaux\seg_test")
print("We have ", len(list(train_dir.glob('*/*.jpg'))), "images in train set.")
print("We have ", len(list(test_dir.glob('*/*.jpg'))), "images in test set.")
class_names = list([item.name for item in train_dir.glob('*')])
print("We have the following classes:", class_names)
image_generator = ImageDataGenerator(rescale=1./255)

train_generator = image_generator.flow_from_directory(train_dir,
                                                      target_size = (150,150),
                                                      batch_size=32,
                                                      class_mode='categorical')

test_generator = image_generator.flow_from_directory(test_dir,
                                                     target_size=(150,150),
                                                     batch_size=32,
                                                     class_mode='categorical')
def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10,10))
    for n in range(10):
        ax = plt.subplot(5,5,n+1)
        plt.imshow(image_batch[n])
        plt.title(np.array(class_names)[label_batch[n]==1][0].title())
        plt.axis('off')
image_batch, label_batch = next(train_generator)
show_batch(image_batch, label_batch)
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
def plot_acc_loss(trained):
    fig, ax = plt.subplots(1, 2, figsize=(15,5))
    ax[0].set_title('loss')
    ax[0].plot(trained.epoch, trained.history["loss"], label="Train loss")
    ax[0].plot(trained.epoch, trained.history["val_loss"], label="Validation loss")
    ax[1].set_title('acc')
    ax[1].plot(trained.epoch, trained.history["accuracy"], label="Train acc")
    ax[1].plot(trained.epoch, trained.history["val_accuracy"], label="Validation acc")
    ax[0].legend()
    ax[1].legend()
# Create the model adding Conv2D. 
model = Sequential([Conv2D(200, (3,3), activation='relu', input_shape=(150, 150, 3)),
                    MaxPool2D(5,5),
                    Conv2D(180, (3,3), activation='relu'),
                    MaxPool2D(5,5),
                    Flatten(),
                    Dense(180, activation='relu'),
                    Dropout(rate=0.25),
                    Dense(2, activation='softmax')])

# Compile model.
# Loss categorical_crossentropy: targets are one-hot encoded.
model.compile(optimizer='adam', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# Fitting the model training with train test and validate test set.
# Callback: early_stopping
trained_CNN = model.fit(train_generator,
                    validation_data  = test_generator,
                    epochs = 40,
                    verbose = 1,
                    callbacks= [early_stopping]);
# Save entire model
model.save('my_model.h5')

# Save weights
model.save_weights('weights_CNN.h5')
plot_acc_loss(trained_CNN)
# Load weights and evaluate model
model.load_weights('weights_CNN.h5')
model_CNN_score = model.evaluate(test_generator)
print("Model CNN Test Loss:", model_CNN_score[0])
print("Model CNN Test Accuracy:", model_CNN_score[1])
# predict with 2 pictures
# Soumettre une image au modèle pour la prédiction
from tensorflow.keras.preprocessing import image

# Charger l'image à prédire
img_path = r'C:\Users\Hadil boussetta\pfe cnn\bateaux\seg_predect\pbe.jpg'
img = image.load_img(img_path, target_size=(150, 150))

# Convertir l'image en tableau numpy
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x /= 255

# Faire la prédiction
prediction = model.predict(x)

# Afficher la prédiction
if prediction[0][0] > 0.5:
    print("bateau entré")
else:
    print("bateau sorti")
plt.imshow(img)