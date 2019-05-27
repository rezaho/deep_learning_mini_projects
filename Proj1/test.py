from utils import *
from argparse import ArgumentParser
from torch.utils.data import DataLoader


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--feature_extractor', type=str, default='100',
        help='Feature extractor architecture to use')
    argparser.add_argument('--target', type=str, default='',
        help='Target classifier architecture')
    argparser.add_argument('--classifier', type=str, default=None,
        help='Digit classifier architecture')
    argparser.add_argument('--num_samples', type=int, default=1000,
        help='Number of training and test samples')
    argparser.add_argument('--optimizer', type=str, default='SGD',
        help='Optimizer to use for updating network weights')
    argparser.add_argument('--batch_size', type=int, default=32,
        help='Size of the batches used at training')
    argparser.add_argument('--lr', type=float, default=0.001,
        help='Learning rate')
    argparser.add_argument('--num_epochs', type=int, default=25,
        help='Number of epochs to train')
    argparser.add_argument('--log_interval', type=int, default=5,
        help='Number of epochs after which to output information about the \
        training progress')
    argparser.add_argument('--setup_type', type=int, default=0,
        help='0->pass both images at the same time; 1->train for digit \
        classification')

    args = argparser.parse_args()

    # Generate training and test data
    train_input, train_target, train_classes, test_input, test_target, \
    test_classes = generate_pair_sets(args.num_samples)

    # Normalize data
    train_input /= 255
    test_input /= 255

    # Generate train and test datasets
    train_dataset = ImageDataset(train_input, train_target, train_classes)
    test_dataset = ImageDataset(test_input, test_target, test_classes)

    # Build train and test loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
        shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
        shuffle=True)

    # Generate network
    if args.setup_type == 0:
        in_shape = '2x14x14'
    else:
        in_shape = '1x14x14'
    model = parse_architecture(in_shape, args.feature_extractor, args.target,
        args.classifier, args.setup_type)

    # Define optimizer
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr)

    # Train the model
    train_loss, train_accuracy, train_aux_accuracy = train(model, train_loader,
        optimizer, vars(args))

    # Test the model
    test_loss, test_accuracy, test_aux_accuracy = test(model, test_loader,
        args.setup_type)

    # Print final results
    to_print = 'Training\tLoss {:.5f}\tAccuracy {:.3f}'.format(train_loss, train_accuracy)
    if train_aux_accuracy > 0:
        to_print += '\tClassification accuracy {:.3f}'.format(train_aux_accuracy)
    print(to_print)

    to_print = 'Test\t\tLoss {:.5f}\tAccuracy {:.3f}'.format(test_loss, test_accuracy)
    if test_aux_accuracy > 0:
        to_print += '\tClassification accuracy {:.3f}'.format(test_aux_accuracy)
    print(to_print)
