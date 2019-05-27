import torch


# Linear Transformation
def linear(input, weight, bias):
    """
    Applies a linear transformation of `X @ W.t()+b`
    
    Inputs:
        - input: mini-batch input X with dim (N,D1)
        - weight: weights matrix with dim (D2, D1)
        - bias : Bias with dim (D2,)
    Output:
        - output: transformed tensor with dim (N,D2)
    """ 
    output = input.mm(weight.t())
    if bias is not None:
        output += bias
    return output

def d_linear(input, weight, output_grad):
    """
    Inputs:
        - weight:      Dim (D2, D1)
        - bias:        Dim (D1,)
        - output_grad: Gradient of output w.r.t. to loss with dim (N, D2)
    Outputs:
        - dldx : Gradient w.r.t. input
        - dldw : Gradient w.r.t. weights
        - dldb : Gradient w.r.t. bias
    """
    dldx = output_grad.mm(weight)
    dldw = output_grad.t().mm(input)
    dldb = output_grad.sum(dim=0)
    return dldx, dldw, dldb



# MSE Loss Implementation
def mse_loss(input, target, reduction='mean'):
    """
    Applies MSE Loss criterion
    """
    if not isinstance(target, torch.FloatTensor):
        target = target.float()
    if reduction=='mean':
        return (input-target).pow(2).mean()
    elif reduction=='sum':
        return (input-target).pow(2).sum()
    else:
        raise ValueError("reduction can get only 'mean' or 'sum' as its attribute!")
        
def d_mse_loss(input, target):
    """
        Returns the gradient of loss w.r.t to input
    """
    if not isinstance(target, torch.FloatTensor):
        target = target.float()
    return 2*(input-target)



# ReLU activation
def relu(input):
    """
    Applies ReLU activation layer
    """
    output = input.clone()
    output[output<0] = 0
    return output

def d_relu(input, output_grad):
    """
    Returns the gradient of loss w.r.t. to input for ReLU activation layer!
    """
    d = (input>0).float()
    return d * output_grad



# Tanh function
def tanh(input):
    """
    Applies Tanh activation layer
    """
    return (1-2*torch.exp(-1*input)/(torch.exp(input)+torch.exp(-1*input)))

def d_tanh(input, output_grad):
    """
    Returns the gradient of loss w.r.t input for Tanh function!
    """
    d = 1 - tanh(input).pow(2)
    return output_grad * d



# Sigmoid
def sigmoid(input):
    """
    Applies Sigmoid activation function!
    """
    return 1 / (1 + torch.exp(-input))

def d_sigmoid(input, output_grad):
    """
    Returns the gradient of loss w.r.t. input in case of Sigmoid activation!
    """
    d = sigmoid(input)
    d = (1 - d) * d
    return d * output_grad



# Softmax
def softmax(input):
    """
    Applies Softmax activation function.
    """
    output = torch.exp(input) / torch.exp(input).sum(1).view(-1,1)
    return torch.max(output, torch.empty(output.shape).fill_(1e-20))

def d_softmax(input, output_grad):
    """
    Returns the gradient of loss w.r.t. input in case of Softmax!
    """
    d = softmax(input)
    d = (1 - d) * d
    return d * output_grad



# CrossEntropy Loss
def cross_entropy(input, target):
    """
    Applies Cross Entropy Loss
    
    Inputs:
        - input: 2D matrix contains the output of a neural network model (N x D)
        - target: 2D matrix contains the one-hot encoded labels! (N x C)
    Output:
        - Cross Entropy Loss
    """
    if len(target.shape)!=2:
        raise ValueError('Target tensor must be a 2D one-hot encoded tensor!')
    if not isinstance(target, torch.FloatTensor):
        target = target.float()
    log_softmax = torch.log(softmax(input))
    return -(target.float()*log_softmax).sum(1).mean()

def d_cross_entropy(input, target):
    """
    Returns the gradient of loss w.r.t. input for Cross Entropy Loss
    """
    if not isinstance(target, torch.FloatTensor):
        target = target.float()
    return (softmax(input)-target)/input.shape[0]