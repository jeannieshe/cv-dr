import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_cf(preds, true, title, class_names=None):
    """
    Creates a confusion matrix with fixed class tick marks and custom styling.
    
    Parameters:
    preds: array-like, list of predictions
    true: array-like, list of true labels
    title: str, title for the figure
    class_names: list of all class labels (e.g., ['class_0', ..., 'class_4'])
        
    Returns:
    fig: matplotlib.figure.Figure object
    """
    # Define fixed label set (default to 0–4)
    if class_names is None:
        class_names = [f'class_{i}' for i in range(5)]  # modify if you have different classes

    labels = list(range(len(class_names)))  # [0, 1, 2, 3, 4]

    # Compute full confusion matrix with fixed labels
    cm = confusion_matrix(true, preds, labels=labels)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(6, 6))

    # Display the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", ax=ax, colorbar=True)

    # Customize tick labels
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_yticklabels(class_names, rotation=90)

    # Add text annotations
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        ax.text(j, i, cm[i, j],
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")

    # Style adjustments
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = disp.im_.colorbar
    if cbar:
        cbar.outline.set_visible(False)

    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title)
    plt.tight_layout()
    # plt.show()

    return fig
