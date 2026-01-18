from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

DEFAULT_IMG_SIZE = (128, 128)
DEFAULT_BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE


from sklearn.utils.class_weight import compute_class_weight
import numpy as np


def compute_class_weights(y, num_classes):
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=y,
    )
    return dict(enumerate(class_weights))


def get_augmentation_layer():
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="data_augmentation",
    )


def load_metadata(metadata_path):
    return pd.read_csv(metadata_path)


def encode_labels(df, label_col="dx"):
    le = LabelEncoder()
    encoded = le.fit_transform(df[label_col])
    return encoded, le


def load_and_preprocess_image(image_path, img_size=DEFAULT_IMG_SIZE):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image


def build_dataset(
    image_paths,
    labels,
    img_size=DEFAULT_IMG_SIZE,
    batch_size=DEFAULT_BATCH_SIZE,
    shuffle=True,
    augment=False,
):
    image_paths = tf.convert_to_tensor(image_paths, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)

    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(image_paths))

    ds = ds.map(
        lambda x, y: (load_and_preprocess_image(x, img_size), y),
        num_parallel_calls=AUTOTUNE,
    )

    if augment:
        augmentation_layer = get_augmentation_layer()
        ds = ds.map(
            lambda x, y: (augmentation_layer(x, training=True), y),
            num_parallel_calls=AUTOTUNE,
        )

    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)

    return ds


def build_image_paths(df, images_dir, image_col="image_id"):
    images_dir = Path(images_dir)
    return df[image_col].apply(lambda x: str(images_dir / f"{x}.jpg")).values


def extract_features_from_dataset(dataset):
    images = []
    labels = []

    for batch_images, batch_labels in dataset:
        batch_images_flat = tf.reshape(batch_images, (batch_images.shape[0], -1))
        images.append(batch_images_flat.numpy())
        labels.append(batch_labels.numpy())

    X = np.vstack(images)
    y = np.concatenate(labels)

    return X, y
