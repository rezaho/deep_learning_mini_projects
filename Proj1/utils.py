import os
import torch
import torch.nn as nn
import torch.optim as optim
from functools import reduce
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import Dataset


# Image dataset generation methods
def mnist_to_pairs(nb, input, target):
    '''
    Reduce MNIST images to a quarter in size and group them in pairs
    input:  nb - number of image pairs to produce
            input - MNIST images
            target - MNIST labels
    output: MNIST image pairs, target label and classes of digits per pair
    '''
    input = torch.functional.F.avg_pool2d(input, kernel_size = 2)
    a = torch.randperm(input.size(0))
    a = a[:2 * nb].view(nb, 2)
    input = torch.cat((input[a[:, 0]], input[a[:, 1]]), 1)
    classes = target[a]
    target = (classes[:, 0] <= classes[:, 1]).long()
    return input, target, classes


def generate_pair_sets(nb):
    '''
    Generate train and test datasets with pairs of downsamples MNIST images
    input:  nb - number of pairs to generate for each dataset
    output: train and test MNIST paired images datasets
    '''
    data_dir = os.environ.get('PYTORCH_DATA_DIR')
    if data_dir is None:
        data_dir = './data'

    train_set = datasets.MNIST(data_dir + '/mnist/', train = True, download = True)
    train_input = train_set.train_data.view(-1, 1, 28, 28).float()
    train_target = train_set.train_labels

    test_set = datasets.MNIST(data_dir + '/mnist/', train = False, download = True)
    test_input = test_set.test_data.view(-1, 1, 28, 28).float()
    test_target = test_set.test_labels

    return mnist_to_pairs(nb, train_input, train_target) + \
           mnist_to_pairs(nb, test_input, test_target)


class ImageDataset(Dataset):
    '''
    Dataset class for MNIST image pairs
    '''

    def __init__(self, data, target, classes, transform=None):
        self.data = data
        self.target = target
        self.classes = classes
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_data = self.data[idx]
        if self.transform:
            sample_data = self.transform(sample_data)
        sample_target = self.target[idx].type(torch.FloatTensor)
        # Transform class labels in categorical data
        sample_class1 = self.classes[idx,0]
        sample_class2 = self.classes[idx,1]
        return sample_data, sample_target, sample_class1, sample_class2


class Flatten(nn.Module):
    '''
    Flatten module used for transitioning from conv to linear layers
    '''

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


# Model parsing methods
def parse_layers(in_shape, architecture, output=None):
    '''
    Parse the layered structure of a part of a network
    input:  in_shape - shape of the input to the module
            architecture - the description of the architecture of the module
            ouput - the number of outputs of the module and the activation
            function to use
    output: the module and the shape of the output of the module
    '''
    # If the list of layers is empty, make the list of layers empty
    if architecture == '':
        layers_desc = []
    else:
        layers_desc = architecture.split(',')

    layers = []
    for layer in layers_desc:
        if '_' in layer:
            # Check input shape
            if len(in_shape) < 3:
                print('Input shape for conv layer is invalid')
                return None
            # Conv layer
            comp = layer.split('_')
            num_channels, kern_size = [int(x) for x in comp[0].split('x')]
            stride = int(comp[1])
            padding = int(comp[2])
            layers.append(nn.Conv2d(in_shape[0], num_channels, kern_size, \
                stride, padding))
            layers.append(nn.ReLU())
            # Change input shape
            new_width = (in_shape[1] - kern_size + padding + 1) // stride
            new_height = (in_shape[2] - kern_size + padding + 1) // stride
            in_shape = (num_channels, new_width, new_height)
        elif 'pool' in layer:
            if len(in_shape) < 3:
                print('Input shape for pool layer is invalid')
            pool_size = int(layer[7:])
            if layer[:3] == 'max':
                # Max pooling layer
                layers.append(nn.MaxPool2d(pool_size))
            else:
                # Average pooling layer
                layers.append(nn.AvgPool2d(pool_size))
            # Change input shape
            in_shape = (in_shape[0], in_shape[1] // 2, in_shape[2] // 2)
        else:
            input_size = reduce(lambda x, y: x * y, in_shape)
            num_out = int(layer)
            # MLP layer
            layers.append(Flatten())
            layers.append(nn.Linear(input_size, num_out))
            layers.append(nn.ReLU())
            # Change input shape
            in_shape = tuple([num_out])

    # Add an output layer in case one is needed
    if output is not None:
        if len(in_shape) > 1:
            layers.append(Flatten())
        in_size = reduce(lambda x, y: x * y, in_shape)
        layers.append(nn.Linear(in_size, output['num_outputs']))
        if output['activation'] == 'Sigmoid':
            layers.append(nn.Sigmoid())
        elif output['activation'] == 'Tanh':
            layers.append(nn.Tanh())
        elif output['activation'] == 'Softmax':
            layers.append(nn.Softmax(dim=1))
        elif output['activation'] == 'ReLU':
            layers.append(nn.ReLU())

    return layers, in_shape


def parse_architecture(input_shape, feature_extractor, target, classifier=None,
    setup_type=0):
    '''
    Parse the string describing the architecture of a network and output a
    model with the corresponding architecture.
    The input shape should be in the form NUM_CHANNELSxWIDTHxHEIGHT
    Layer descriptions are separated by comma.
    Fully-connected layers are represented as NUMHIDDENNEURONS.
    Conv layers are represented as NUMCHANNELSxKERNSIZE_STRIDE_PADDING.
    num_outputs is a list of dictionaries describing the number of outputs and
    activation function to use for different predictions
    Pooling layers are represented as POOLTYPEPOOLSIZE
    input:  input_shape - the shape of the input of the network
            feature_extractor - the architecture of the feature extractor
            target - the architecture of the target classifier
            classifier - the architecture of the digit classifier
            setup_type - whether we do direct classification or digit classification
    output: a network with the architecture described
    '''
    in_shape = tuple([int(x) for x in input_shape.split('x')])

    # Build feature extractor
    feature_layers, in_shape = parse_layers(in_shape, feature_extractor)

    # Build the target classifier
    if setup_type == 0:
        target_output = {'num_outputs': 1, 'activation': 'Sigmoid'}
        target_layers, _ = parse_layers(in_shape, target, target_output)
        if classifier is not None:
            # Construct digit classifiers
            class_output = {'num_outputs': 10, 'activation': 'Softmax'}
            classifier1_layers, _ = parse_layers(in_shape, classifier, class_output)
            classifier2_layers, _ = parse_layers(in_shape, classifier, class_output)
    else:
        class_output = {'num_outputs': 10, 'activation': 'Softmax'}
        classifier_layers, _ = parse_layers(in_shape, classifier, class_output)

    # Generate model
    class MyModel(nn.Module):
        '''
        Class used to define a model for digit comparison
        '''

        def __init__(self):
            super(MyModel, self).__init__()
            self.feature_extractor = nn.ModuleList(feature_layers)
            # Decide on the number of classification modules based on the setup type
            if setup_type == 0:
                self.target = nn.ModuleList(target_layers)
                if classifier is not None:
                    self.classifier1 = nn.ModuleList(classifier1_layers)
                    self.classifier2 = nn.ModuleList(classifier2_layers)
            else:
                self.classifier = nn.ModuleList(classifier_layers)

        def forward(self, x):
            # Extract features
            for layer in self.feature_extractor:
                x = layer(x)

            # Make classification based on setup type
            if setup_type == 0:
                # Predict target
                aux = x
                for layer in self.target:
                    aux = layer(aux)
                target = aux

                # Predict digits
                if classifier is not None:
                    # Classify first digit
                    aux = x
                    for layer in self.classifier1:
                        aux = layer(aux)
                    class1 = aux
                    # Classify second digit
                    aux = x
                    for layer in self.classifier2:
                        aux = layer(aux)
                    class2 = aux
                    return target, class1, class2
            else:
                for layer in self.classifier:
                    x = layer(x)
                target = x

            return target, None, None

    return MyModel()


# Train and test routines
def train(model, train_loader, optimizer, args):
    '''
    Train the model
    input:  model - model to be trained
            train_loader - loader for training data
            optimizer - optimizer used for updating model weights
            args - arguments to be used for training
    output: final training loss, accuracy and digit classification accuracy
    '''
    model.train()
    for epoch in range(1, args['num_epochs'] + 1):
        for batch_idx, (data, target, class1, class2) in enumerate(train_loader):
            # Prepare optimizer
            optimizer.zero_grad()

            # Compute loss based on training method used
            if args['setup_type'] == 0:
                output = model(data)
                loss = F.binary_cross_entropy(output[0], target.view(-1,1))
                if args['classifier'] is not None:
                    loss += F.cross_entropy(output[1], class1) + \
                            F.cross_entropy(output[2], class2)
            else:
                # Classify each image
                output1, _, _ = model(data[:,0].unsqueeze(1))
                output2, _, _ = model(data[:,1].unsqueeze(1))
                # Compute loss
                loss = F.cross_entropy(output1, class1) + \
                        F.cross_entropy(output2, class2)
            # Backpropagate
            loss.backward()
            # Optimize network
            optimizer.step()
        # Ouput training progress
        if epoch % args['log_interval'] == 0:
            loss, accuracy, aux_accuracy = test(model, train_loader, args['setup_type'])
            to_print = 'Epoch {}/{}\tLoss {:.5f}\tAccuracy {:.3f}'.format(epoch,
                args['num_epochs'], loss, accuracy)
            if aux_accuracy > 0:
                to_print += '\tClassification accuracy {:.3f}'.format(aux_accuracy)
            print(to_print)

    # Evaluate model on training data
    loss, accuracy, aux_accuracy = test(model, train_loader, args['setup_type'])

    return loss, accuracy, aux_accuracy


def test(model, test_loader, setup_type=0):
    '''
    Test the model
    input:  model - the model to be tested
            test_loader - loader for test data
            setup_type - type of setup to use for solving the task
    output: test loss, accuracy and digit classification accuracy
    '''
    model.eval()
    loss, corr, aux_corr = 0, 0, 0
    with torch.no_grad():
        for (data, target, class1, class2) in test_loader:
            if setup_type == 0:
                output = model(data)
                loss += F.binary_cross_entropy(output[0], target.view(-1,1))
                pred = (output[0] > 0.5).type(torch.FloatTensor)
                corr += pred.eq(target.view(-1,1)).sum().item()
                # Also compute the accuracy on the auxiliary classification task
                if output[1] is not None:
                    loss += F.cross_entropy(output[1], class1) + \
                                F.cross_entropy(output[2], class2)
                    pred1 = output[1].argmax(dim=1)
                    pred2 = output[2].argmax(dim=1)
                    aux_corr += pred1.eq(class1).sum().item() + \
                                    pred2.eq(class2).sum().item()
            else:
                # Classify each image
                output1, _, _ = model(data[:,0].unsqueeze(1))
                output2, _, _ = model(data[:,1].unsqueeze(1))
                # Compute loss
                loss += F.cross_entropy(output1, class1) + \
                        F.cross_entropy(output2, class2)
                # Compute accuracies
                pred1 = output1.argmax(dim=1)
                pred2 = output2.argmax(dim=1)
                result = (pred1 <= pred2).float()
                corr += result.eq(target).sum().item()
                aux_corr += pred1.eq(class1).sum().item() + \
                            pred2.eq(class2).sum().item()

    # Compute the overall loss and accuraccies
    loss /= len(test_loader.dataset)
    accuracy = corr / len(test_loader.dataset)
    aux_accuracy = aux_corr / len(test_loader.dataset) / 2

    return loss.item(), accuracy, aux_accuracy
