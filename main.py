from dataset import validation_set, validation_loader, train_loader
from utils import load_model, create_confusion_matrix, plot_random_predictions
from consts import MODEL_SAVE_PATH
from model import loss_fn, accuracy_fn, optimizer, model
from test import test_model
from train import train_model

# epochs = 3
# for epoch in range(epochs):
#     train_loss, train_accuracy = train_model(model, train_loader, loss_fn, accuracy_fn, optimizer)
#     val_loss, val_accuracy = test_model(model, validation_loader, loss_fn, accuracy_fn)
#     print(
#         f"Train accuracy: {train_accuracy:.3f} | Train loss: {train_loss:.3f} | Test accuracy: {val_accuracy:.3f} | "
#         f"Test loss: {val_loss:.3f}")

loaded_model = load_model(MODEL_SAVE_PATH)

plot_random_predictions(loaded_model, validation_set)
create_confusion_matrix(loaded_model, validation_loader, validation_set.targets)

if __name__ == '__main__':
    pass
