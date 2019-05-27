import torch
import helper as H
import nn
from argparse import ArgumentParser

def main():
    arg_parser = ArgumentParser()
    # Hyper parameters for the dataset generation
    arg_parser.add_argument('--nb_training', type=int, default=1000,
        help='number of training samples to generate')
    arg_parser.add_argument('--nb_test', type=int, default=1000,
        help='number of test samples to generate')
    
    # Type of Loss function
    arg_parser.add_argument('--loss', type=str, default='CrossEntropyLoss',
        help='Loss criterion to use. \n- Options:\n\t- MSELoss \n\t- CrossEntropyLoss')
    
    # Optimizer type and its hyper parameters
    arg_parser.add_argument('--optim', type=str, default='Adam',
        help='Optimizer to train.\n-Options: \n\t- SGD\n\t- Adam')
    arg_parser.add_argument('--beta1', type=float, default=0.9,
        help='Beta 1 hyper parameter (Only when using Adam optim.)')
    arg_parser.add_argument('--beta2', type=float, default=0.999,
        help='Beta 2 hyper parameter (Only when using Adam optim.)')
    arg_parser.add_argument('--eps', type=float, default=1e-8,
        help='Epsilon hyper parameter (Only when using Adam optim.)')
    
    arg_parser.add_argument('--weight_decay', type=float, default=0,
        help='Weight decay (for L2 penalty)')
    arg_parser.add_argument('--lr', type=float, default=None,
        help='Learning Rate for Optimizer')
    
    # Training Hyper parameters
    arg_parser.add_argument('--model', type=str, default='Custom',
        help='The module for building the model.\nOptions:\n\t1. Custom\n\t2. Sequential')
    arg_parser.add_argument('--nb_epochs', type=int, default=500,
        help='Number of epochs for training phase.')
    arg_parser.add_argument('--batch_size', type=int, default=100,
        help='The size of mini-batch in the training phase.')

    args = arg_parser.parse_args()
    
    # Setting learning rate
    if args.lr is None and args.optim=='Adam':
        lr = 2e-3
    elif args.lr is None and args.optim=='SGD':
        lr = 2e-1
    else:
        lr = args.lr
    
    # Setting PyTorch Autograd Machinery Off
    torch.set_grad_enabled(False)
    
    # Creating the training and test set
    train_input, train_target = H.generate_dataset(args.nb_training)
    test_input, test_target = H.generate_dataset(args.nb_test)

    # Normalizing the mean and variance
    train_input, test_input = H.normalize(train_input, test_input)

    # One hot encoding
    train_target,_ = H.one_hot_encoding(train_target)
    test_target,_ = H.one_hot_encoding(test_target)

    # creating a model 
    if args.model == 'Sequential':
        model = nn.Sequential(nn.Linear(2,128), nn.ReLU(), nn.Linear(128,2))
        
    elif args.model == 'Custom':
        class Net(nn.Module):
            def __init__(self):
                self.fc1 = nn.Linear(2, 128)
                self.fc2 = nn.Linear(128, 2)
                self.relu = nn.ReLU()
                super(Net, self).__init__()
            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.fc2(x)
                return x
            def __call__(self, input):
                return self.forward(input)
        model = Net()
    else:
        raise ValueError('model parameter can only get "Custom" or "Sequential" as an input.')
    
    # Training the model using training set
    print('\n Training Model Created By {} Module...'.format(args.model))
    print('\t Optimizer --> {}\tLearning Rate --> {}'.format(args.optim, lr))
    H.train_model(model, train_input, train_target, lr=lr, batch_size=args.batch_size, nb_epochs=args.nb_epochs, 
                  weight_decay=args.weight_decay, betas=(args.beta1, args.beta2), eps=args.eps, 
                  criterion=args.loss, optimizer=args.optim)
    
    # Calculating the train and test error: Custom Module
    nb_errors = H.compute_nb_errors(model, train_input, train_target)
    print('\tTraining Set: \tNumber of errors: {}\tError Rate: {:,.1f}%'.format(nb_errors, nb_errors/train_input.shape[0]*100.0))
    nb_errors = H.compute_nb_errors(model, test_input, test_target)
    print('\tTest Set: \t\tNumber of errors: {}\tError Rate: {:,.1f}%\n'.format(nb_errors, nb_errors/test_input.shape[0]*100.0))

    
if __name__ == '__main__':
    main()
