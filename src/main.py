import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

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
    choices=["cnn_iter", "cnn", "knn", "tl"],
    help="Which model to run",
)
args = parser.parse_args()


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "HAM10000_images"
METADATA_PATH = DATA_DIR / "HAM10000_metadata.csv"
OUTPUT_DIR = PROJECT_ROOT / "output"


def make_experiment_name(**kwargs):
    return "_".join([f"{k}{v}" for k, v in kwargs.items()])


IMG_SIZE = (96, 96)
BATCH_SIZE = 64
VAL_SPLIT = 0.2
EPOCHS = 100
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

if args.model == "cnn_iter":
    print("Running CNN with checkpointing, resume, and milestone evaluation...")

    import tensorflow as tf
    import json

    cnn_root = OUTPUT_DIR / "cnn"
    checkpoint_dir = cnn_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history_path = checkpoint_dir / "history.json"

    def save_history(new_history, path):
        if path.exists():
            with open(path, "r") as f:
                old = json.load(f)
        else:
            old = {}

        for k, v in new_history.items():
            old.setdefault(k, []).extend(v)

        with open(path, "w") as f:
            json.dump(old, f)

    def plot_history_from_json(history_path, output_dir):
        with open(history_path, "r") as f:
            history = json.load(f)

        output_dir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(10, 4))

        # Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history["accuracy"], label="train")
        plt.plot(history["val_accuracy"], label="val")
        plt.title("Accuracy")
        plt.legend()

        # Loss
        plt.subplot(1, 2, 2)
        plt.plot(history["loss"], label="train")
        plt.plot(history["val_loss"], label="val")
        plt.title("Loss")
        plt.legend()

        plt.tight_layout()
        plt.savefig(output_dir / "training_history.png")
        plt.close()

    def write_summary_txt(history_path, epoch, output_dir):
        with open(history_path, "r") as f:
            history = json.load(f)

        idx = epoch - 1

        with open(output_dir / "summary.txt", "w") as f:
            f.write(f"Epoch: {epoch}\n")
            f.write(f"Image size: {IMG_SIZE}\n")
            f.write(f"Batch size: {BATCH_SIZE}\n")
            f.write(f"Train accuracy: {history['accuracy'][idx]:.4f}\n")
            f.write(f"Val accuracy: {history['val_accuracy'][idx]:.4f}\n")
            f.write(f"Train loss: {history['loss'][idx]:.4f}\n")
            f.write(f"Val loss: {history['val_loss'][idx]:.4f}\n")

    latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)

    if latest_ckpt:
        print(f"Resuming from checkpoint: {latest_ckpt}")
        model = tf.keras.models.load_model(latest_ckpt)
        initial_epoch = int(latest_ckpt.split("_")[-1].split(".")[0])
    else:
        model = build_cnn_model(
            input_shape=(*IMG_SIZE, 3),
            num_classes=len(class_names),
        )
        initial_epoch = 0

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_dir / "epoch_{epoch:03d}.keras"),
        save_weights_only=False,
        save_freq="epoch",
        verbose=1,
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        initial_epoch=initial_epoch,
        class_weight=class_weights,
        callbacks=[checkpoint_cb],
    )

    save_history(history.history, history_path)
    plot_history_from_json(history_path, cnn_root)

    MILESTONE_EPOCHS = [10, 25, 50, 75, 100]

    for e in MILESTONE_EPOCHS:
        ckpt_path = checkpoint_dir / f"epoch_{e:03d}.keras"
        if not ckpt_path.exists():
            continue

        print(f"Evaluating checkpoint at epoch {e}...")

        model_e = tf.keras.models.load_model(ckpt_path)

        exp_dir = cnn_root / f"epochs{e}_img{IMG_SIZE[0]}_bs{BATCH_SIZE}"
        exp_dir.mkdir(parents=True, exist_ok=True)

        save_confusion_matrix(
            model_e,
            val_ds,
            class_names,
            exp_dir,
        )

        write_summary_txt(
            history_path,
            epoch=e,
            output_dir=exp_dir,
        )

    print("CNN training, checkpointing, and evaluation complete.")

elif args.model == "cnn":
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
