import torch
from nn import Parameter, Variable
from collections import defaultdict

class Optimizer(object):
    """
    The base class for Optimizers
    """
    def __init__(self, parameters, others):
        if not isinstance(others, dict):
            raise TypeError("Other parameters must be a dictionary!")
        self.others = others
        self.params = list()
        params_input = list(parameters)
        if len(params_input)==0:
            raise TypeError("Optimizer got an empty list of parameters!")
        self.params = params_input  
        self.state = defaultdict(dict)
        
    def step(self):
        raise NotImplementedError
        

            
class SGD(Optimizer):
    """
    SGD optimizer
    
    Arguments:
    |
    |-- parameters \t: list of parameters in nn.Module for optimization
    |-- learning_rate\t:learning rate
    |-- weght_decay\t: implements L2 penalty
    |
    """
    def __init__(self, parameters, learning_rate=1e-2, weight_decay=0):
        if learning_rate<=0.0:
            raise ValueError("Learning rate must be strictly greater than zero!")
        defaults_dict = dict(learning_rate=learning_rate, weight_decay=weight_decay)
        super(SGD, self).__init__(parameters, defaults_dict)
        
    def step(self):
        lr = self.others['learning_rate']
        weight_decay = self.others['weight_decay']
        for p in self.params:
            # continuing in case "p" has no gradient field or the gradient is set off
            if p.grad_v is None or p.grad_f==False:
                continue
            grad_p = p.grad_v.data
            if weight_decay!=0:
                grad_p.add_(weight_decay, p.data)
            
            # Updating Coefficients
            p.data.add_(-lr, grad_p)

            

class Adam(Optimizer):
    """
    Adam Optimizer

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
    |
    |-- parameters     : list of parameters in nn.Module for optimization
    |-- learning_rate  : learning rate
    |-- betas          : beta values for exponential moving avg. and exp moving average of squared gradients
    |-- eps            : epsilon value to avoid exploding the gradient steps
    |-- weght_decay    : implements L2 penalty
    |
    """

    def __init__(self, parameters, learning_rate=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if learning_rate<=0.0:
            raise ValueError("Invalid learning rate: {} \t Learning rate should be strictly greater than zero!".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta_1: {} \t Beta should be in range [0,1]".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta_2: {} \t Beta should be in range [0,1]".format(betas[1]))
        if eps<=0.0:
            raise ValueError("Invalid epsilon: {} \t Epsilon should be strictly Greater than zero!".format(eps))

        defaults_dic = dict(learning_rate=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay)
        
        super(Adam, self).__init__(parameters, defaults_dic)

    def step(self):
        """
        Performs a Single Optim. Step!
        """
        for p in self.params:
            # continuing in case "p" has no gradient field or the gradient is set off
            if p.grad_v is None or p.grad_f==False:
                continue
            # Initializing hyper parameters
            lr = self.others['learning_rate']
            w_decay = self.others['weight_decay']
            beta1, beta2 = self.others['betas']
            eps = self.others['eps']
            
            # Initializing the state
            if len(self.state[p]) == 0:
                # Number of steps: t
                self.state[p]['t'] = 0
                # Exponential moving average of gradients
                self.state[p]['avg'] = torch.zeros_like(p.data)
                # Exponential moving average of squared gradients
                self.state[p]['avg_sq'] = torch.zeros_like(p.data)
            
            # Updating the step
            self.state[p]['t'] += 1
            
            # Updating the gradient field
            grad = p.grad_v.data
            if w_decay != 0:
                grad.add_(w_decay, p.data)

            # Updating
            self.state[p]['avg'].mul_(beta1).add_(1 - beta1, grad)
            self.state[p]['avg_sq'].mul_(beta2).add_(1 - beta2, grad.pow(2))
            
            # Getting the state
            state = self.state[p]
            t = state['t']
            
            # Getting exp. moving average and exp. moving average od squared
            m, v = self.state[p]['avg'], self.state[p]['avg_sq']
            # Correcting Bias
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            
            # Updating the Coefficient
            p.data.addcdiv_(-lr, m_hat, (v_hat.sqrt() + eps))
