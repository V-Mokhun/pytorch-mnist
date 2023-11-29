import random
import torch
import matplotlib.pyplot as plt
from model import MNISTModel
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
from dataset import NUM_CLASSES, classnames


def save_model(state_dict, save_path):
    torch.save(obj=state_dict, f=save_path)


def load_model(save_path):
    loaded_model = MNISTModel()
    loaded_model.load_state_dict(torch.load(f=save_path))
    return loaded_model


def create_confusion_matrix(model, loader, targets):
    preds = []
    model.eval()
    with torch.inference_mode():
        for X, y in loader:
            logit = model(X)
            pred = torch.softmax(logit, dim=1).argmax(dim=1)
            preds.append(pred)
    preds_tensor = torch.cat(preds)

    confmat = ConfusionMatrix(num_classes=NUM_CLASSES, task='multiclass')
    confmat_tensor = confmat(preds=preds_tensor,
                             target=targets)

    fig, ax = plot_confusion_matrix(
        conf_mat=confmat_tensor.numpy(),
        class_names=classnames,
        figsize=(10, 7)
    )
    plt.show()

def plot_random_predictions(model, validation_set):
    test_samples = []
    test_labels = []
    for sample, label in random.sample(list(validation_set), k=9):
        test_samples.append(sample)
        test_labels.append(label)

    model.eval()
    with torch.inference_mode():
        pred_probs = []
        for sample in test_samples:
            sample = torch.unsqueeze(sample, dim=0)
            pred_logits = model(sample)
            pred_prob = torch.softmax(pred_logits, dim=1)

            pred_probs.append(pred_prob)

        pred_probs = torch.stack(pred_probs).squeeze(dim=1)
        pred_labels = pred_probs.argmax(dim=1).squeeze(dim=0)

    plt.figure(figsize=(9, 9))
    nrows = 3
    ncols = 3
    for i, sample in enumerate(test_samples):
        plt.subplot(nrows, ncols, i + 1)
        plt.imshow(sample.squeeze(), cmap="gray")

        pred_label = classnames[pred_labels[i]]
        truth_label = classnames[test_labels[i]]
        title_text = f"Pred: {pred_label} | Truth: {truth_label}"

        if pred_label == truth_label:
            plt.title(title_text, fontsize=10, c="g")
        else:
            plt.title(title_text, fontsize=10, c="r")
        plt.axis(False);

    plt.show()
