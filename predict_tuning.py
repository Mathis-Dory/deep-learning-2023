import logging
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
from keras.utils import set_random_seed
from sklearn.metrics import confusion_matrix

current_dir = os.path.dirname(__file__)
set_random_seed(42)

img_height, img_width = 64, 64
batch_size = 64
train_path = "./data/train_images/"
val_path = "./data/val_images/"
test_path = "./data/test_images/"
df_train = pd.read_csv('./data/train.csv')
df_val = pd.read_csv('./data/val.csv')


def init() -> None:
    logging.info(f"Amount of images in train_images: {len(os.listdir('data/train_images'))}")
    logging.info(f"Amount of images in val_images: {len(os.listdir('data/val_images'))}")
    logging.info(f"Amount of images in test_images: {len(os.listdir('data/test_images'))}")
    logging.debug(f"Shape of train DF: {df_train.shape}")
    logging.debug(f"Shape of val DF: {df_val.shape}")
    logging.info(f"Labels distribution for training: {df_train['Class'].value_counts()}")
    logging.info(f"Labels distribution for validation: {df_val['Class'].value_counts()}")

    # Plot 4 random images from training set with their
    random_indices = df_train.sample(n=4).index
    plt.figure(figsize=(20, 15))
    for i, idx in enumerate(random_indices):
        image_name = df_train.loc[idx, 'Image']
        image_class = df_train.loc[idx, 'Class']

        image_path = os.path.join(train_path, image_name)
        image = Image.open(image_path)

        plt.subplot(2, 2, i + 1)
        plt.imshow(image)
        plt.title(f"{image_name} | Class: {image_class}", pad=10, fontsize=10)  # Add padding to the title
        plt.axis('off')

    # Adjust subplot parameters
    plt.subplots_adjust(hspace=0.3, wspace=0.9)  # Increase horizontal and vertical spacing

    plt.show()

def structure_data(df: pd.DataFrame, dir_type: str) -> None:
    working_dir = 'working_dir'
    os.makedirs(working_dir, exist_ok=True)
    for index, row in df.iterrows():
        image_filename = row['Image']
        class_label = str(row['Class'])  # Convert class label to string
        class_dir = f"{working_dir}/{dir_type}/{class_label}"
        os.makedirs(class_dir, exist_ok=True)
        src = f'{dir_type}/{image_filename}'
        dest = f'{class_dir}/{image_filename}'
        shutil.copyfile(src, dest)
    logging.info(f"Finished structuring {dir_type} data")


def preprocess() -> (DirectoryIterator, DirectoryIterator, DirectoryIterator, float, float):
    working_train = os.path.join(current_dir, 'working_dir', 'data', 'train_images')
    working_val = os.path.join(current_dir, 'working_dir', 'data','val_images')
    test_dir = os.path.join(current_dir, 'test_dir')
    train_generator = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.2,
        horizontal_flip=False,
        fill_mode='nearest'
    )
    val_generator = ImageDataGenerator(rescale=1. / 255, )
    test_generator = ImageDataGenerator(rescale=1. / 255)
    train_gen = train_generator.flow_from_directory(
        working_train,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
    )
    batch = train_gen.next()

    # Extract the first image from the batch
    first_image = batch[0][0]

    # Display the first normalized image
    logging.debug(f"Normalized image: {first_image}")
    val_gen = val_generator.flow_from_directory(
        working_val,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_gen = test_generator.flow_from_directory(
        test_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode=None,  # Set to None as we're only predicting, not training
        shuffle=False  # Ensure that the order of predictions matches file order
    )

    return train_gen, val_gen, test_gen


def retrain_model() -> None:
    filepath = f'models/{model_name}/{model_name}.h5'
    model = load_model('models/search_tuning/best_model_tuned.h5')
    model.summary()
    checkpoint = ModelCheckpoint(
        filepath=filepath,
        monitor="val_accuracy",
        mode="max",
        verbose=1,
        save_best_only=True,
    )
    early = EarlyStopping(patience=6, restore_best_weights="True", monitor="val_loss")
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)
    callbacks = [checkpoint, early, reduce_lr]
    history = model.fit(
        train_gen,
        epochs=128,
        steps_per_epoch=len(train_gen),
        validation_data=val_gen,
        validation_steps=len(val_gen),
        verbose=1,
        batch_size=batch_size,
        shuffle=True,
        callbacks=callbacks,
    )
    val_loss, val_acc = \
        model.evaluate(val_gen, steps=len(df_val))

    print('val_loss:', val_loss)
    print('val_acc:', val_acc)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(f"models/{model_name}/loss.png", format="png", dpi=1200)
    plt.figure()

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig(f"models/{model_name}/accuracy.png", format="png", dpi=1200)
    plt.figure()
    plt.show()

    plot_confusion_matrix(model, val_gen, steps=len(df_val))


def plot_confusion_matrix(model, generator, steps) -> None:
    generator.shuffle = False

    # Get true class labels
    true_classes = generator.classes
    class_labels = list(generator.class_indices.keys())

    # Predict classes
    predictions = model.predict(generator, steps=steps)
    predicted_classes = np.argmax(predictions, axis=1)

    # Generate the confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes, normalize='true')

    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f"models/{model_name}/confusion_matrix.png", format="png", dpi=1200)
    plt.show()


def predict() -> None:
    model = load_model(f"models/{model_name}/{model_name}.h5")
    predictions = model.predict(test_generator, steps=len(test_generator), verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    class_labels = list(train_gen.class_indices.keys())
    image_names = []
    predicted_classes_list = []

    # Prepare data for DataFrame
    for i, file_name in enumerate(test_generator.filenames):
        image_name = os.path.basename(file_name)
        predicted_class = class_labels[predicted_classes[i]]
        image_names.append(image_name)
        predicted_classes_list.append(predicted_class)

    data = {'Image': image_names, 'Class': predicted_classes_list}
    df_preds = pd.DataFrame(data)

    # Save DataFrame to CSV

    df_preds.to_csv(f"models/{model_name}/submission-{model_name}.csv", index=False, columns=['Image', 'Class'])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init()
    if not os.path.exists("working_dir/train_images") or not os.path.exists("working_dir/val_images"):
        logging.critical("working_dir or test_dir do not exist. Please run structure_data() first, then run this "
                         "script again.")
        exit(0)
    else:

        model_name = "mod_tuned"
        os.makedirs(f"models/{model_name}", exist_ok=True)
        train_gen, val_gen, test_generator = preprocess()
        retrain_model()
        predict()
