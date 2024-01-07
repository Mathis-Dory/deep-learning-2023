import logging
import os
import shutil
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv2D
from keras.layers import Dense, Flatten, BatchNormalization, Add, Input
from keras.models import load_model, Model
from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
from keras.src.callbacks import ReduceLROnPlateau
from keras.src.initializers.initializers import HeNormal
from keras.src.layers import Activation, GlobalAveragePooling2D
from keras.src.optimizers import Adam
from keras.utils import set_random_seed, plot_model
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay

set_random_seed(42)

# Set up global variables
img_height, img_width, channels = 64, 64, 3
batch_size = 64
labels = 100
train_path = "./data/train_images/"
val_path = "./data/val_images/"
test_path = "./data/test_images/"
model_name = "resnet"
optimizer = Adam()

# Load datasets
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


def preprocess() -> (DirectoryIterator, DirectoryIterator, DirectoryIterator, float, float):
    working_train = 'working_dir/train_images'
    working_val = 'working_dir/val_images'
    test_dir = 'test_dir'
    train_generator = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=10,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.15,
        brightness_range=[0.9, 1.1],
        shear_range=0.05,
        horizontal_flip=True,
        vertical_flip=False,
        channel_shift_range=10
    )
    val_generator = ImageDataGenerator(
        rescale=1. / 255,

    )
    train_gen = train_generator.flow_from_directory(
        working_train,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
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
        shuffle=False  # Ensure that the order of predictions matches files order
    )

    return train_gen, val_gen, test_generator


def block(input, filter, initializer, strides=1) -> Any:
    # Save the input value for a skip connection
    x_skip = input

    x = Conv2D(filter, (3, 3), strides=strides, padding='same', kernel_initializer=initializer)(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filter, (3, 3), padding='same', kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)

    # If the stride is not 1 or the number of filters in the input doesn't match the number of filters in the
    # convolutional layers, we will have a shape issue when adding the input to the output because the shapes won't
    # match. To solve this, we apply a 1x1 convolution to the input to change the number of filters.
    if strides != 1 or input.shape[3] != filter:
        x_skip = Conv2D(filter, (1, 1), strides=strides, padding='same', kernel_initializer=initializer)(x_skip)
        x_skip = BatchNormalization(axis=3)(x_skip)

    x = Add()([x, x_skip])
    x = Activation('relu')(x)
    return x


def residual_layers(x, filter, initializer) -> Any:
    # We create the different groups of residual layers
    # 3 groups of residual layers with 3 blocks in each group
    layers = [3, 3, 3]
    for i in range(len(layers)):
        for j in range(layers[i]):
            if i > 0 and j == 0:
                # First layer of the second and third group with stride 2 (need to the skip connection to have the
                # same shape)
                x = block(x, filter, initializer, strides=2)
            else:
                x = block(x, filter, initializer)
        # Double the number of filters for each group
        filter *= 2
    return x


def create_resnet(input_shape, labels, initializer) -> Model:
    inputs = Input(shape=input_shape)
    # Initial conv layer, the stride is 1 (conv filter movement is 1 pixel) and padding is same to ensure that the
    # output shape matches the input shape
    x = Conv2D(16, (3, 3), strides=(1, 1), padding='same', kernel_initializer=initializer)(inputs)
    # Batch normalization is applied to normalize the inputs of each layer to improve training speed and stability
    # Batch normalization normalizes the activations of the previous layer for each given neuron, across the batch.
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Residual layers are used to increase the depth of the network
    x = residual_layers(x, 16, initializer)

    # Reduce each feature map to a single number by taking the average of the numbers in that feature map.
    x = GlobalAveragePooling2D()(x)
    # Flatten the pooled feature map to a single vector
    x = Flatten()(x)
    # Final dense layer with softmax activation. It converts the output to probability scores for each class.
    output = Dense(labels, activation='softmax')(x)

    return Model(inputs, output)


def create_model() -> None:
    model = create_resnet((img_width, img_height, channels), labels, HeNormal())
    # Compile model
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    # Model summary
    model.summary()

    # Save model summary to file
    plot_model(model, to_file=f"models/{model_name}/model.png", show_shapes=True, show_layer_names=True)
    filepath = f"models/{model_name}/{model_name}.hdf5"
    checkpoint = ModelCheckpoint(
        filepath=filepath,
        monitor="val_accuracy",
        mode="max",
        verbose=1,
        save_best_only=True,
    )
    early = EarlyStopping(monitor='val_loss', mode='min', patience=50, restore_best_weights=True, verbose=1)
    # Reduce the learning rate when the validation loss stops improving.
    # Decreasing the learning rate helps the model to fine-tune and find minima.
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=0.00001)
    callbacks = [checkpoint, early, reduce_lr]
    history = model.fit(
        train_gen,
        epochs=200,
        steps_per_epoch=len(train_gen),
        validation_data=val_gen,
        validation_steps=len(val_gen),
        verbose=1,
        batch_size=batch_size,
        shuffle=True,
        callbacks=callbacks,
    )
    # Plot training history
    plot_training_history(history)

    # Confusion matrix and F1 score
    plot_confusion_matrix_and_score()


def plot_training_history(history):
    plt.plot(history.history["accuracy"], color="red")
    plt.plot(history.history["val_accuracy"], color="purple")
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.savefig(f"models/{model_name}/accuracy.png", format="png", dpi=300)
    plt.figure()
    plt.show()

    plt.plot(history.history["loss"], color="green")
    plt.plot(history.history["val_loss"], color="blue")
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.savefig(f"models/{model_name}/loss.png", format="png", dpi=300)
    plt.figure()
    plt.show()


def plot_confusion_matrix_and_score():
    best_model = load_model(f"models/{model_name}/{model_name}.hdf5")

    y_true = np.concatenate([val_gen.next()[1] for _ in range(len(val_gen))])
    y_pred = best_model.predict(val_gen, steps=len(val_gen))
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_true, axis=1)

    # Calculate the confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes, normalize='true')

    # Plotting a normalized confusion matrix with improved label visibility
    fig, ax = plt.subplots(figsize=(64, 64))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(len(val_gen.class_indices)))
    disp.plot(include_values=False, cmap='viridis', ax=ax, xticks_rotation='vertical')
    ax.set_title('Normalized Confusion Matrix')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(f"models/{model_name}/normalized_confusion_matrix_improved.png", dpi=300)
    plt.show()

    # Evaluate the model to get validation loss and accuracy
    val_loss, val_acc = best_model.evaluate(val_gen, steps=len(df_val))
    print('Validation Loss:', val_loss)
    print('Validation Accuracy:', val_acc)

    # Calculate the F1 score
    f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')
    print(f'F1 Score: {f1}')


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

    # Save predictions to CSV

    df_preds.to_csv(f"models/{model_name}/submission-{model_name}.csv", index=False, columns=['Image', 'Class'])



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init()
    # structure_data(df_train, 'train_images')  # Run only once for the whole project
    # structure_data(df_val, 'val_images')  # Run only once for the whole project
    # prepare_test()  # Run only once for the whole project
    if not os.path.exists("working_dir/train_images") or not os.path.exists("working_dir/val_images"):
        logging.critical("working_dir or test_dir do not exist. Please run structure_data() first, then run this "
                         "script again.")
        exit(0)
    else:
        train_gen, val_gen, test_generator = preprocess()
        create_model()
        predict()
