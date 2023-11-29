import torch


def train_model(model, train_loader, loss_fn, accuracy_fn, optimizer):
    model.train()
    train_loss = 0
    train_accuracy = 0
    for batch, (X, y) in enumerate(train_loader):
        logits = model(X)
        preds = torch.softmax(logits, dim=1).argmax(dim=1)

        loss = loss_fn(logits, y)
        train_loss += loss
        train_accuracy += accuracy_fn(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_accuracy /= len(train_loader)
    train_loss /= len(train_loader)

    return train_loss, train_accuracy
