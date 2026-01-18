import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

from transfer_model import build_transfer_model, fine_tune_model

from image_preprocessing import (
    compute_class_weights,
    load_metadata,
    encode_labels,
    build_image_paths,
    build_dataset,
    extract_features_from_dataset,
)

from cnn_model import (
    build_cnn_model,
    plot_training_history,
    save_confusion_matrix,
)

from knn_model import (
    build_knn_model,
    plot_knn_results,
    save_knn_confusion_matrix,
)


parser = argparse.ArgumentParser(description="Skin Cancer Classification")
parser.add_argument(
    "--model",
    type=str,
    required=True,
    choices=["cnn", "knn", "tl"],
    help="Which model to run",
)
args = parser.parse_args()


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "HAM10000_images"
METADATA_PATH = DATA_DIR / "HAM10000_metadata.csv"
OUTPUT_DIR = PROJECT_ROOT / "output"


IMG_SIZE = (96, 96)
BATCH_SIZE = 64
VAL_SPLIT = 0.2
EPOCHS = 75
RANDOM_STATE = 42
if args.model == "tl":
    IMG_SIZE = (128, 128)
    EPOCHS = 20


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

class_weights = compute_class_weights(
    train_labels,
    num_classes=len(class_names),
)

with open(OUTPUT_DIR / "weights.txt", "w") as f:
    f.write("Class weights:\n")
    for i, w in class_weights.items():
        f.write(f"{class_names[i]}: {w:.2f}\n")

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
    print("Running CNN...")

    model = build_cnn_model(
        input_shape=(*IMG_SIZE, 3),
        num_classes=len(class_names),
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weights,
    )

    cnn_output_dir = OUTPUT_DIR / "cnn"

    plot_training_history(history, cnn_output_dir)
    save_confusion_matrix(model, val_ds, class_names, cnn_output_dir)

    print(f"Results saved to {cnn_output_dir}")

elif args.model == "knn":
    print("Running kNN classifier...")

    train_ds = build_dataset(
        train_image_paths,
        train_labels,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
        augment=False,
    )

    val_ds = build_dataset(
        val_image_paths,
        val_labels,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
        augment=False,
    )

    X_train, y_train = extract_features_from_dataset(train_ds)
    X_val, y_val = extract_features_from_dataset(val_ds)

    knn_output_dir = OUTPUT_DIR / "knn"

    knn_model, knn_history = build_knn_model(
        X_train, y_train, X_val, y_val, n_neighbors=5
    )

    plot_knn_results(knn_history, knn_output_dir)
    save_knn_confusion_matrix(knn_model, X_val, y_val, class_names, knn_output_dir)

    print(f"Results saved to {knn_output_dir}")

elif args.model == "tl":
    print("Running transfer learning model (MobileNetV2)...")

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    model, base_model = build_transfer_model(
        input_shape=(*IMG_SIZE, 3),
        num_classes=len(class_names),
    )

    tl_output_dir = OUTPUT_DIR / "transfer_learning"

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=[early_stop],
    )

    plot_training_history(history, tl_output_dir)
    save_confusion_matrix(model, val_ds, class_names, tl_output_dir)

    print(f"Results saved to {tl_output_dir}")
