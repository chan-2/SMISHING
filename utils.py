import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from collections import deque
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from tqdm import tqdm
def calculate_test_loss_and_accuracy(model, device, loss_function, test_data_loader, X_on_the_fly_function=None):
    model.eval()
    correct = 0
    with torch.inference_mode():
        average_test_loss = 0
        for test_data in tqdm(test_data_loader, position=0, leave=False, desc="Calculating Test Loss & Accuracy"):
            test_X, test_y = test_data
            if X_on_the_fly_function is not None:
                test_X = X_on_the_fly_function(test_X)
            test_X = test_X.to(device)
            test_y = test_y.to(device)
            test_y_output = model(test_X)
            test_y_prediction = torch.round(test_y_output)
            correct += accuracy_score(test_y.cpu().detach(), test_y_prediction.cpu().detach(), normalize=False)
            test_loss = loss_function(test_y_output, test_y)
            average_test_loss += test_loss
        average_test_loss /= len(test_data_loader.dataset)
    return average_test_loss, correct / len(test_data_loader.dataset)


def print_learning_progress(epoch, train_loss, test_loss, accuracy=None):
    progress_string = "\nepoch: {}"\
                      "\ntrain loss: {}"\
                      "\ntest loss : {}".format(epoch, train_loss, test_loss)
    if accuracy is not None:
        progress_string += "\naccuracy: {}".format(accuracy)
    print(progress_string)


def train_loop(train_data_set, test_data_set, epochs, model, device, batch_size, loss_function, optimizer,
               print_interval, X_on_the_fly_function=None,
               collate_fn=torch.utils.data.default_collate, test_first=False, shuffle=True, print_tsne=True,
               drop_last=True, print_graph=True, print_matrix=False, model_save_path=None):

    train_data_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, drop_last=drop_last)
    test_data_loader = DataLoader(test_data_set, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, drop_last=drop_last)
    last_accuracy = 0
    if print_graph:
        y_train_losses = deque()
        y_test_losses = deque()
        x_epochs = deque()
    if test_first:
        last_accuracy = print_progress(train_data_loader, test_data_loader, model, device, 0, loss_function, 0, X_on_the_fly_function)

    for epoch in range(1, epochs+1):
        average_train_loss = 0
        for train_data in tqdm(train_data_loader, position=0, leave=False, desc="Training"):
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
            average_train_loss, average_test_loss, last_accuracy = print_progress(train_data_loader, test_data_loader, model, device, epoch, loss_function, average_train_loss, X_on_the_fly_function)
            if print_graph:
                y_train_losses.append(average_train_loss.detach().cpu().numpy())
                y_test_losses.append(average_test_loss.detach().cpu().numpy())
                x_epochs.append(epoch)
    if print_matrix:
        print_confusion_matrix(model=model, data_loader=test_data_loader, X_on_the_fly_function=X_on_the_fly_function)
    if print_tsne:
        print_tsne_model_output(model=model, data_loader=test_data_loader, X_on_the_fly_function=X_on_the_fly_function)
    if print_graph:
        print_training_graph(x_epochs, y_train_losses, y_test_losses)
    if model_save_path is not None:
        torch.save(model, model_save_path)
    return last_accuracy


def print_progress(train_data_loader, test_data_loader, model, device, epoch, loss_function, average_train_loss, X_on_the_fly_function=None):
    average_train_loss /= len(train_data_loader.dataset)
    average_test_loss, accuracy = calculate_test_loss_and_accuracy(model, device, loss_function, test_data_loader, X_on_the_fly_function)
    print_learning_progress(epoch, average_train_loss, average_test_loss, accuracy)
    return average_train_loss, average_test_loss, accuracy

def print_confusion_matrix(model, data_loader, X_on_the_fly_function=None):
    y_preds = deque()
    ys = deque()
    model.eval()
    with torch.inference_mode():
        for (X, y) in tqdm(data_loader, position=0, leave=False, desc="Creating Confusion Matrix"):
            if X_on_the_fly_function is not None:
                X = X_on_the_fly_function(X)
            y_pred = torch.round(model(X))
            y_preds.append(y_pred.cpu())
            ys.append(y.cpu())
    ys = torch.cat(list(ys), dim=0).cpu()
    y_preds = torch.cat(list(y_preds), dim=0).cpu()
    print(classification_report(ys, y_preds, target_names=["Not Spam", "Spam"]))
def print_tsne_model_output(model, data_loader, X_on_the_fly_function=None):
    ys = deque()
    X_embeddings = deque()
    # Predict label
    model.eval()
    with torch.inference_mode():
        for (X, y) in tqdm(data_loader, position=0, leave=False, desc="Creating T-SNE"):
            if X_on_the_fly_function is not None:
                X = X_on_the_fly_function(X)
            _, X_embedding = model(X, True)
            ys.append(y)
            X_embeddings.append(X_embedding)
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
def print_training_graph(x_epochs, y_train_losses, y_test_losses):
    plt.plot(x_epochs, y_train_losses, label='Train Loss')
    plt.plot(x_epochs, y_test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Learning Curve')
    plt.legend()
    plt.show()

def print_last_layer_weights(model):
    parameters = list(model.parameters())[-2].detach().cpu().numpy().reshape(-1)
    index = np.arange(0, parameters.shape[0])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(index, parameters, color='blue')
    ax.set_xlabel('Index')
    ax.set_ylabel('Weight')
    ax.set_title('Visualization of the last layer')
    plt.show()

def get_device_name_agnostic():
    return "cuda" if torch.cuda.is_available() else "cpu"

