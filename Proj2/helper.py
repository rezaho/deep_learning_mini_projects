import torch
import math
import nn
import optim


def generate_dataset(nb):
    """
    Generates dataset!
    
    Inputs:
        - nb :      Number of samples to create.
    
    Outputs:
        - input_  : train/test samples with 2 features (nb, 2)
        - target_ : labels (nb,)
    """
    input_ = torch.empty(nb,2).uniform_(0,1)
    target_ = torch.where((input_.pow(2).sum(dim=1))<=math.sqrt(1/(2*math.pi)), torch.ones(nb), torch.zeros(nb)).long()
    return input_, target_


def normalize(train, test):
    """
    Normalize the mean and variance
    
    Inputs:
        - train : training input features (N, D)
        - test  : test input features (M, D)
        
    Outputs:
        - train : normalized train set
        - test  : normalized test set (based on train meand and std)
    """
    mean, std = train.mean(dim=0), train.std(dim=0)
    train = (train - mean)/ std
    test = (test - mean)/std
    return train, test


# One-hot-encoding
def one_hot_encoding(input):
    """
    One-hot encoder
    
    Inputs:
        - input : 1D tensor containing labels (N,)
    Outputs:
        - output: one-hot encoded tensor (N, C)
        - classes: all unique class in order (C,)
    """
    if len(input.shape)>1 and input.shape[1]>1:
        raise ValueError("Tensor to be encoded, should have only one dimension or the second dimension should have size of one!")
    classes = input.unique()
    N = input.shape[0]
    C = classes.shape[0]
    output = torch.zeros(N,C).long()
    output[torch.arange(N), input] = 1
    return output, classes


def train_model(model, train_input, train_target, lr=1e-1, batch_size=100, nb_epochs=500,
                betas=(0.9, 0.999), eps=1e-8, weight_decay=0, criterion='MSELoss', optimizer='SGD'):
    """
    Training the model
    
    Arguments:
    |
    |-- model       : An instance of the Module class
    |-- train_input : Training set (N, D)
    |-- train_target: Training labels (N,)
    |-- lr          : Learning rate for optimizer
    |-- batch_size  : mini batch size used in training
    |-- nb_epochs   : number of epochs
    |-- criterion   : Type of loss function to be used. Available options: 1.'MSELoss', 2.'CrossEntropyLoss'
    |-- optimizer   : Type of optimizer to be used. Available options: 1.'SGD'
    |
    
    Outputs:
    |
    |-- loss    : the values of loss
    |
    """
    if lr is None:
        raise ValueError('Learning rate cannot be None.')
    
    # Creating the Loss module
    if criterion=='MSELoss':
        criterion = nn.MSELoss()
    elif criterion=='CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError("Criterion's value can either be 'MSELoss' or 'CrossEntropyLoss'!")
    
    # Creating the optimizer object
    if optimizer=='SGD':
        optimizer = optim.SGD(model.parameters(), learning_rate=lr, weight_decay=weight_decay)
    elif optimizer=='Adam':
        optimizer = optim.Adam(model.parameters(), learning_rate=lr, weight_decay=weight_decay, betas=betas, eps=eps)
    else:
        raise ValueError('Optimizer can only get "SGD" as its value!')
    
    # Starting epochs
    
    for e in range(nb_epochs):
        for b in range(0, train_input.shape[0], batch_size):
            output = model(train_input[b : b+batch_size])
            loss = criterion(output, train_target[b:b+batch_size]) 
            
            model.zero_gradients()
            loss.backward_run()
            optimizer.step()
            
        print('\t Steps: {} / {} ...'.format(e+1, nb_epochs) , end="\r")
    print('\t Steps: {} / {} ...'.format(e+1, nb_epochs))
    return loss

def compute_nb_errors(model, data_input, data_target):
    """
    Computes the number of errors using the trained model
    """
    batch_size = 100
    nb_errors = 0
    for b in range(0,data_input.shape[0], batch_size):
        output = model(data_input[b: b+batch_size]).max(1)[1]
        real = data_target[b: b+batch_size].max(1)[1]
        nb_errors += (output!=real).sum().item()
    return nb_errors
