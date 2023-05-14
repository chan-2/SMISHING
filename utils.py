import torch
from torch.utils.data import DataLoader
from collections import deque
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def calculate_test_loss(model, device, loss_function, test_data_loader, X_on_the_fly_function=None):
    model.eval()
    with torch.inference_mode():
        average_test_loss = 0
        for test_data in test_data_loader:
            test_X, test_y = test_data
            if X_on_the_fly_function is not None:
                test_X = X_on_the_fly_function(test_X)
            test_X = test_X.to(device)
            test_y = test_y.to(device)
            test_y_prediction = model(test_X)
            test_loss = loss_function(test_y_prediction, test_y)
            average_test_loss += test_loss
        average_test_loss /= len(test_data_loader.dataset)
    return average_test_loss

def calculate_accuracy(model, test_data_loader, X_on_the_fly_function=None):
    # Predict label
    correct = 0
    model.eval()
    with torch.inference_mode():
        for (X, y) in test_data_loader:
            if X_on_the_fly_function is not None:
                X = X_on_the_fly_function(X)
            y_pred = torch.round(model(X))
            correct += accuracy_score(y.cpu().detach(), y_pred.cpu().detach(), normalize=False)
    return correct / len(test_data_loader.dataset)

def print_tsne_model_output(model, data_loader):
    y_preds = deque()
    ys = deque()
    X_embeddings = deque()
    # Predict label
    model.eval()
    with torch.inference_mode():
        for (X, y) in data_loader:
            X = model.embed_texts(X)
            y_pred, X_embedding = model(X, True)
            y_preds.append(y_pred)
            ys.append(y)
            X_embeddings.append(X_embedding)
        y_preds = torch.cat(list(y_preds), dim=0).cpu()
        ys = torch.cat(list(ys), dim=0).cpu()
        X_embeddings = torch.cat(list(X_embeddings), dim=0).cpu()

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    transformed_data = tsne.fit_transform(X_embeddings)

    # Visualize t-SNE output
    not_spam_index = (ys == 0).squeeze()
    spam_index = (ys == 1).squeeze()
    plt.scatter(transformed_data[not_spam_index, 0], transformed_data[not_spam_index, 1], c="green", label="Not Spam")
    plt.scatter(transformed_data[spam_index, 0], transformed_data[spam_index, 1], c="red", label="Spam")
    plt.legend()
    plt.show()

    return


def print_learning_progress(epoch, train_loss, test_loss, accuracy=None):
    progress_string = "\nepoch: {}"\
                      "\ntrain loss: {}"\
                      "\ntest loss : {}".format(epoch, train_loss, test_loss)
    if accuracy is not None:
        progress_string += "\naccuracy: {}".format(accuracy)
    print(progress_string)


def train_loop(train_data_set, test_data_set, epochs, model, device, batch_size, loss_function, optimizer,
               print_interval, accuracy_function=None, X_on_the_fly_function=None,
               collate_fn=torch.utils.data.default_collate, test_first=False, shuffle=True, print_tsne=True,
               drop_last=True, print_graph=True):

    train_data_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, drop_last=drop_last)
    test_data_loader = DataLoader(test_data_set, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, drop_last=drop_last)
    last_accuracy = 0
    if print_graph:
        y_train_losses = deque()
        y_test_losses = deque()
        x_epochs = deque()
    if test_first:
        last_accuracy = print_progress(train_data_loader, test_data_loader, model, device, 0, loss_function, 0, accuracy_function, X_on_the_fly_function)

    for epoch in range(1, epochs+1):
        average_train_loss = 0
        for train_data in train_data_loader:
            model.train()
            X, y = train_data
            if X_on_the_fly_function is not None:
                X = X_on_the_fly_function(X)
            X = X.to(device)
            y = y.to(device)

            y_prediction = model(X)

            loss = loss_function(y_prediction, y)
            average_train_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if print_interval <= 0:
            continue
        if epoch % print_interval == 0:
            average_train_loss, average_test_loss, last_accuracy = print_progress(train_data_loader, test_data_loader, model, device, epoch, loss_function, average_train_loss, accuracy_function, X_on_the_fly_function)
            if print_graph:
                y_train_losses.append(average_train_loss.detach().cpu().numpy())
                y_test_losses.append(average_test_loss.detach().cpu().numpy())
                x_epochs.append(epoch)
    if print_tsne:
        print_tsne_model_output(model=model, data_loader=test_data_loader)
    if print_graph:
        print_training_graph(x_epochs, y_train_losses, y_test_losses)
    return last_accuracy


def print_progress(train_data_loader, test_data_loader, model, device, epoch, loss_function, average_train_loss, accuracy_function=None, X_on_the_fly_function=None):
    average_train_loss /= len(train_data_loader.dataset)
    average_test_loss = calculate_test_loss(model, device, loss_function, test_data_loader, X_on_the_fly_function)
    if accuracy_function is None:
        print_learning_progress(epoch, average_train_loss, average_test_loss)
        return average_train_loss, average_test_loss
    else:
        accuracy = accuracy_function(model, test_data_loader, X_on_the_fly_function)
        print_learning_progress(epoch, average_train_loss, average_test_loss, accuracy)
        return average_train_loss, average_test_loss, accuracy


def print_training_graph(x_epochs, y_train_losses, y_test_losses):
    plt.plot(x_epochs, y_train_losses, label='Train Loss')
    plt.plot(x_epochs, y_test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Learning Curve')
    plt.legend()
    plt.show()
def get_device_name_agnostic():
    return "cuda" if torch.cuda.is_available() else "cpu"

