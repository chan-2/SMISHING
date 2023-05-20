import torch.cuda
import random
from data.load_mecab_data import *
from baseline_model import *
from utils import *
import itertools
from transformers import logging

if __name__ == '__main__':
    # ignore repetitive error
    logging.set_verbosity_error()
    # set seed
    seed = 2020
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # load data
    train_dataset, test_dataset = load_mecab_dataset(random_state=seed)
    # get device
    device = get_device_name_agnostic()
    # set hyper parameters
    lr_list = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]
    epochs_list = [5, 10] # The epochs should be in increasing order
    batch_size_list = [4, 8, 12, 16]
    print_interval = 5
    loss_function = torch.nn.BCELoss()
    # storage for search results
    accuracies = []
    parameter_sets = []
    best_parameter_set = None
    best_accuracy = 0
    # grid search
    for lr, batch_size in itertools.product(lr_list, batch_size_list):
        # load model & set optimizer
        model = BaselineModel(input_size=1024, hidden_size=1024).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        prev_epochs = 0
        # train loop
        for epochs in epochs_list:
            current_parameter_set = {"lr": lr, "epoch": epochs, "batch_size": batch_size}
            current_accuracy = train_loop(train_data_set=train_dataset, test_data_set=test_dataset,
                               epochs=epochs - prev_epochs, model=model, device=device,
                               batch_size=batch_size, loss_function=loss_function, optimizer=optimizer,
                               print_interval=print_interval, accuracy_function=calculate_accuracy,
                               X_on_the_fly_function=model.tokenize_texts, test_first=False,
                               shuffle=False, print_tsne=False, drop_last=False, print_graph=False, print_matrix=False)
            prev_epochs = epochs
            accuracies.append(current_accuracy)
            parameter_sets.append(current_parameter_set)
            print("Parameter set: {}, Accuracy: {}".format(current_parameter_set, current_accuracy))
            if best_accuracy < current_accuracy:
                best_accuracy = current_accuracy
                best_parameter_set = current_parameter_set

    for accuracy, parameter_set in zip(accuracies, parameter_sets):
        print("Parameter set: {}, Accuracy: {}".format(parameter_set, accuracy))
    print("Best Parameter set: {}, Accuracy: {}".format(best_parameter_set, best_accuracy))





