import torch


def test_model(model, validation_loader, loss_fn, accuracy_fn):
    model.eval()
    with torch.inference_mode():
        val_accuracy = 0
        val_loss = 0
        for X, y in validation_loader:
            val_logits = model(X)
            val_preds = torch.softmax(val_logits, dim=1).argmax(1)

            val_loss += loss_fn(val_logits, y)
            val_accuracy += accuracy_fn(val_preds, y)
        val_accuracy /= len(validation_loader)
        val_loss /= len(validation_loader)

    return val_loss, val_accuracy
