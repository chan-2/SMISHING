import torch.cuda
import random
from data.load_mecab_data import *
from model import *
from utils import *

if __name__ == '__main__':
    # set seed
    seed = 2020
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # load data
    train_dataset, test_dataset = load_mecab_dataset(random_state=seed)
    # load model
    device = get_device_name_agnostic()
    model = Model(input_size=768, hidden_size=512).to(device)
    # set hyper parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)
    loss_function = torch.nn.BCELoss()
    epochs = 100
    print_interval = 1
    batch_size = 16
    # train loop
    train_loop(train_data_set=train_dataset, test_data_set=test_dataset, epochs=epochs, model=model, device=device,
               batch_size=batch_size, loss_function=loss_function, optimizer=optimizer, print_interval=print_interval,
               accuracy_function=calculate_accuracy, X_on_the_fly_function=model.embed_texts, test_first=True, shuffle=False,
               print_tsne=True, drop_last=False, print_graph=True)


