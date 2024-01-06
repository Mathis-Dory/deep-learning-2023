import logging
import os
import shutil

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.initializers import VarianceScaling
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, BatchNormalization, GlobalMaxPooling2D, GlobalAveragePooling2D, \
    Flatten
from keras.optimizers.legacy import Adam
from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
from keras.regularizers import l2
from keras.utils import set_random_seed
from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch

set_random_seed(42)

img_height, img_width = 64, 64
batch_size = 32
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
        rotation_range=20,
        brightness_range=[0.8, 1.2],
        zoom_range=0.3,
        vertical_flip=True,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
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


class CNNHyperModel(HyperModel):
    def __init__(self, img_height, img_width):
        self.img_height = img_height
        self.img_width = img_width

    def build(self, hp) -> Sequential:
        model = Sequential()
        model.add(Conv2D(filters=hp.Choice('filters_1', values=[32,64, 128]), activation='relu', padding=hp.Choice('padding_1', values=['same', 'valid']),
                         kernel_size=hp.Choice('kernel_1', values=[3, 5]), input_shape=(img_width, img_height, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=hp.Choice('strides_1', values=[1, 2])))
        model.add(Dropout(rate=hp.Float('dropout_1', min_value=0, max_value=0.3, step=0.1)))

        model.add(Conv2D(filters=hp.Choice('filters_2', values=[64, 128, 256]), activation='relu',
                         kernel_size=hp.Choice('kernel_2', values=[3, 5]), padding=hp.Choice('padding_2', values=['same', 'valid'])))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=hp.Choice('strides_2', values=[1, 2])))
        model.add(Dropout(rate=hp.Float('dropout_2', min_value=0, max_value=0.3, step=0.1)))

        model.add(Conv2D(filters=hp.Choice('filters_3', values=[64, 128, 256, 512]), activation='relu',
                         kernel_size=hp.Choice('kernel_3', values=[3, 5]), padding=hp.Choice('padding_3', values=['same', 'valid'])))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=hp.Choice('strides_3', values=[1, 2])))
        model.add(Dropout(rate=hp.Float('dropout_3', min_value=0, max_value=0.3, step=0.1)))

        model.add(BatchNormalization())
        pooling_choice = hp.Choice('pooling', values=['global_max_pooling', 'global_avg_pooling'])
        if pooling_choice == 'global_max_pooling':
            model.add(GlobalMaxPooling2D())
        elif pooling_choice == 'global_avg_pooling':
            model.add(GlobalAveragePooling2D())

        model.add(Flatten())
        if hp.Boolean('Dense_sup'):  # This will either be True or False during tuning
            model.add(Dense(units=hp.Choice('dense_units_sup', values=[128, 256, 512]), activation='relu',
                            kernel_initializer=VarianceScaling(),
                            kernel_regularizer=l2(), activity_regularizer=l2()))
            model.add(Dropout(rate=hp.Float('dropout_sup', min_value=0.0, max_value=0.5, step=0.1)))

        model.add(Dense(units=hp.Choice('dense_final', values=[128, 256, 512]), activation='relu',
                        kernel_initializer=VarianceScaling(),
                        kernel_regularizer=l2(), activity_regularizer=l2()))
        model.add(Dropout(rate=hp.Choice(f'dropout_final', values=[0.0, 0.3, 0.5])))

        # Output layer
        model.add(Dense(100, activation="softmax"))

        model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model


def find_best(train_gen, val_gen):
    tuner = RandomSearch(
        hypermodel,
        objective='val_accuracy',
        max_trials=15,
        directory='keras_tuner_dir',
        seed=42,
        project_name='tunner'
    )
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    callbacks = [early_stopping]
    tuner.search(train_gen,
                 epochs=20,
                 validation_data=val_gen,
                 callbacks=callbacks)

    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.save(f'models/{model_name}/best_model_tuned.h5')
    best_model.summary()


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

        model_name = "cnn_tuning"
        os.makedirs(f"models/{model_name}", exist_ok=True)
        train_gen, val_gen, test_generator = preprocess()
        hypermodel = CNNHyperModel(img_height=img_height, img_width=img_width)
        find_best(train_gen, val_gen)
