from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
from pathlib import Path


def build_knn_model(X_train, y_train, X_val, y_val, n_neighbors=5):
    print(f"Training kNN with n_neighbors={n_neighbors}...")

    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")
    knn.fit(X_train, y_train)

    train_pred = knn.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)

    val_pred = knn.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)

    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")

    history = {
        "train_accuracy": [train_acc],
        "val_accuracy": [val_acc],
        "train_loss": [1 - train_acc],
        "val_loss": [1 - val_acc],
        "n_neighbors": n_neighbors,
    }

    return knn, history


def plot_knn_results(history, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    bars = plt.bar(
        ["Train", "Validation"],
        [history["train_accuracy"][0], history["val_accuracy"][0]],
        color=["blue", "orange"],
    )
    plt.title(f'kNN Accuracy (k={history["n_neighbors"]})')
    plt.ylabel("Accuracy")
    plt.ylim([0, 1])

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.3f}",
            ha="center",
            va="bottom",
        )

    plt.subplot(1, 2, 2)
    bars = plt.bar(
        ["Train", "Validation"],
        [history["train_loss"][0], history["val_loss"][0]],
        color=["blue", "orange"],
    )
    plt.title(f'kNN Loss (k={history["n_neighbors"]})')
    plt.ylabel("Loss")
    plt.ylim([0, 1])

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.3f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(output_dir / "knn_results.png", dpi=150)
    plt.close()

    with open(output_dir / "knn_metrics.txt", "w") as f:
        f.write(f"kNN Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Number of neighbors: {history['n_neighbors']}\n")
        f.write(f"Training accuracy: {history['train_accuracy'][0]:.4f}\n")
        f.write(f"Validation accuracy: {history['val_accuracy'][0]:.4f}\n")
        f.write(f"Training loss: {history['train_loss'][0]:.4f}\n")
        f.write(f"Validation loss: {history['val_loss'][0]:.4f}\n")


def save_knn_confusion_matrix(model, X_val, y_val, class_names, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(X_val)

    cm = confusion_matrix(y_val, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    plt.title(f"kNN Confusion Matrix (k={model.n_neighbors})")
    plt.tight_layout()
    plt.savefig(output_dir / "knn_confusion_matrix.png", dpi=150)
    plt.close()

    with open(output_dir / "knn_class_metrics.txt", "w") as f:
        f.write("Per-class metrics for kNN\n")
        f.write("=" * 60 + "\n")

        for i, class_name in enumerate(class_names):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - (tp + fn + fp)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            f.write(f"\nClass: {class_name}\n")
            f.write(f"  True Positives: {tp}\n")
            f.write(f"  False Negatives: {fn}\n")
            f.write(f"  False Positives: {fp}\n")
            f.write(f"  Precision: {precision:.3f}\n")
            f.write(f"  Recall: {recall:.3f}\n")
            f.write(f"  F1-Score: {f1:.3f}\n")
