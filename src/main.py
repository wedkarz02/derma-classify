import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split

from image_preprocessing import (
    load_metadata,
    encode_labels,
    build_image_paths,
    build_dataset,
)

from cnn_model import (
    build_cnn_model,
    plot_training_history,
    save_confusion_matrix,
)

parser = argparse.ArgumentParser(description="Skin Cancer Classification")
parser.add_argument(
    "--model",
    type=str,
    required=True,
    choices=["cnn"],
    help="Which model to run",
)
args = parser.parse_args()


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "HAM10000_images"
METADATA_PATH = DATA_DIR / "HAM10000_metadata.csv"
OUTPUT_DIR = PROJECT_ROOT / "output"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
VAL_SPLIT = 0.2
EPOCHS = 10
RANDOM_STATE = 42


df = load_metadata(METADATA_PATH)
labels, label_encoder = encode_labels(df)
class_names = label_encoder.classes_


lesion_ids = df["lesion_id"].unique()

train_lesions, val_lesions = train_test_split(
    lesion_ids,
    test_size=VAL_SPLIT,
    random_state=RANDOM_STATE,
)

train_df = df[df["lesion_id"].isin(train_lesions)]
val_df = df[df["lesion_id"].isin(val_lesions)]

train_image_paths = build_image_paths(train_df, IMAGES_DIR)
val_image_paths = build_image_paths(val_df, IMAGES_DIR)

train_labels = label_encoder.transform(train_df["dx"])
val_labels = label_encoder.transform(val_df["dx"])


train_ds = build_dataset(
    train_image_paths,
    train_labels,
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    augment=True,
)

val_ds = build_dataset(
    val_image_paths,
    val_labels,
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    augment=False,
)


if args.model == "cnn":
    print("Running baseline CNN...")

    model = build_cnn_model(
        input_shape=(*IMG_SIZE, 3),
        num_classes=len(class_names),
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
    )

    cnn_output_dir = OUTPUT_DIR / "cnn"

    plot_training_history(history, cnn_output_dir)
    save_confusion_matrix(model, val_ds, class_names, cnn_output_dir)

    print(f"Results saved to {cnn_output_dir}")
