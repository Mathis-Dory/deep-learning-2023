import logging
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from keras import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.initializers import VarianceScaling
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
from keras.src.optimizers import Adam
from keras.utils import set_random_seed

set_random_seed(42)

img_height, img_width = 64, 64
batch_size = 128
train_path = "./data/train_images/"
val_path = "./data/val_images/"
test_path = "./data/test_images/"
model_name = "cnn_basic"
optimizer = Adam()


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

    # Plot 4 random images from training set
    random_indices = df_train.sample(n=4).index
    plt.figure(figsize=(20, 15))
    for i, idx in enumerate(random_indices):
        image_name = df_train.loc[idx, 'Image']
        image_class = df_train.loc[idx, 'Class']
        image_path = os.path.join(train_path, image_name)
        image = Image.open(image_path)

        plt.subplot(2, 2, i + 1)
        plt.imshow(image)
        plt.title(f"{image_name} | Class: {image_class}", pad=10, fontsize=12)
        plt.axis('off')

    plt.subplots_adjust(hspace=0.3, wspace=0.9)
    os.makedirs(f"models/{model_name}", exist_ok=True)
    plt.savefig(f"models/{model_name}/random_images.png", format="png", dpi=300)
    plt.show()


# Structure the folders as follows:
# data
# |
# |___train
# |      |___class_1
# |      |___class_2
# |
# |___validation
# |      |___class_1
# |      |___class_2
def structure_data(df: pd.DataFrame, dir_type: str) -> None:
    working_dir = 'working_dir'
    os.makedirs(working_dir, exist_ok=True)
    for index, row in df.iterrows():
        image_filename = row['Image']
        class_label = str(row['Class'])  # Convert class label to string
        class_dir = f"{working_dir}/{dir_type}/{class_label}"
        os.makedirs(class_dir, exist_ok=True)
        src = os.path.join('data', dir_type, image_filename)
        dest = os.path.join(class_dir, image_filename)
        shutil.copyfile(src, dest)
    logging.info(f"Finished structuring {dir_type} data")


def preprocess() -> (DirectoryIterator, DirectoryIterator, DirectoryIterator, float, float):
    working_train = 'working_dir/train_images'
    working_val = 'working_dir/val_images'
    test_dir = 'test_dir'
    train_generator = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=10,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True,
        vertical_flip=False,
    )
    val_generator = ImageDataGenerator(
        rescale=1. / 255,

    )
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

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode=None,  # Set to None as we're only predicting, not training
        shuffle=False  # Ensure that the order of predictions matches file order
    )

    return train_gen, val_gen, test_generator


def create_model() -> None:
    model = Sequential()
    model.add(Conv2D(256, (3, 3), padding='same', input_shape=(img_height, img_width, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Dropout(0.2))

    model.add(BatchNormalization())
    model.add(GlobalAveragePooling2D())
    model.add(Flatten())

    model.add(Dense(128, activation="relu", kernel_initializer=VarianceScaling(),
                    kernel_regularizer="l2", activity_regularizer="l2"))
    model.add(Dropout(0.3)),
    model.add(Dense(100, activation="softmax"))

    model.summary()

    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    logging.info(f"Model compiled {val_gen.class_indices}")

    filepath = f"models/{model_name}/{model_name}.hdf5"
    checkpoint = ModelCheckpoint(
        filepath=filepath,
        monitor="val_accuracy",
        mode="max",
        verbose=1,
        save_best_only=True,
    )
    early = EarlyStopping(patience=10, restore_best_weights="True", monitor="val_loss")
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)
    callbacks = [checkpoint, early, reduce_lr]
    history = model.fit(
        train_gen,
        epochs=100,
        steps_per_epoch=len(train_gen),
        validation_data=val_gen,
        validation_steps=len(val_gen),
        verbose=1,
        batch_size=batch_size,
        shuffle=True,
        callbacks=callbacks,
    )
    val_loss, val_acc = model.evaluate(val_gen, steps=len(df_val))

    print('val_loss:', val_loss)
    print('val_acc:', val_acc)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']




def prepare_test() -> None:
    test_dir = 'test_dir'
    os.makedirs(test_dir, exist_ok=True)
    test_images = os.path.join(test_dir, 'test_images')
    os.mkdir(test_images)
    test_list = os.listdir('data/test_images')
    for image in test_list:
        src = os.path.join('data/test_images', image)
        dst = os.path.join(test_images, image)
        shutil.copyfile(src, dst)


def predict() -> None:
    model = load_model(f"models/{model_name}/{model_name}.hdf5")
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
    # structure_data(df_train, 'train_images')
    # structure_data(df_val, 'val_images')
    # prepare_test()
    if not os.path.exists("working_dir/train_images") or not os.path.exists("working_dir/val_images"):
        logging.critical("working_dir or test_dir do not exist. Please run structure_data() first, then run this "
                         "script again.")
        exit(0)
    else:
        os.makedirs(f"models/{model_name}", exist_ok=True)
        train_gen, val_gen, test_generator = preprocess()
        create_model()
        predict()


