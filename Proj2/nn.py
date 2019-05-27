import torch
import functional as F
from collections import OrderedDict

class Module(object):
    """
    The Base Class for Modules
    """
    def __init__(self):
        self.parameters_ = OrderedDict()
        self.modules_ = OrderedDict()
        # Adding Parameters
        self.add_parameters()
        # Adding Modules
        self.add_modules()
        
    def forward(self, *input ):
        raise NotImplementedError
    
    def backward(self, *gradwrtoutput):
        raise NotImplementedError
            
    def parameters(self):
        """
        Returns an iterator over module parameters.
        This is typically passed to an optimizer.
        """
        for name, parameter in self.parameters_with_names():
            yield parameter

    def parameters_with_names(self):
        """
        Returns an iterator over module parameters, yielding both the
        name of the parameter as well as the parameter itself.
        """
        memo = set()
        for module in self.modules():
            for k, v in module.parameters_.items():
                if v is None or v in memo:
                    continue
                memo.add(v)
                yield k, v     

    def modules(self, memo=None):
        """
        Returns an iterator over all modules in the network, yielding
        both the name of the module as well as the module itself.
        """
        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            yield self
            for name, module in self.modules_.items():
                if module is None:
                    continue
                for m in module.modules(memo):
                    yield m
                    
    def add_parameters(self):
        """
        This function adds all available parameters to the instance!
        """
        for k, v in self.__dict__.items():
            if isinstance(v, Parameter):
                self.register_parameter(k, v)
    
    def register_parameter(self, name, param):
        """
        Register a specific parameter by having its name and object
        """
        self.parameters_[name] = param
        
    def add_modules(self):
        """
        Adds all available modules to the instance
        """
        memo = set(self.modules_.values())
        for k, v in self.__dict__.items():
            if isinstance(v, Module) and v not in memo:
                memo.add(v)
                self.register_module(k,v)
        
    def register_module(self, name, module):
        """
        Register a specific module by having its name and object
        """
        self.modules_[name] = module
        
    def zero_gradients(self):
        """
        Set all the gradients to zero recursively!
        """
        for p in self.parameters():
            if p.grad_v is not None:
                p.grad_v.zero_()

                
                
                
class Linear(Module):
    """
    Fully Connected Linear Layer
    """
    def __init__(self, in_feat, out_feat, bias=True):
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.weight = Parameter(torch.empty(out_feat, in_feat).normal_(std=1e-6), grad_v=torch.full((out_feat, in_feat), 0.0))
        if bias:
            self.bias = Parameter(torch.empty(out_feat).normal_(std=1e-6), grad_v=torch.full((out_feat,), 0.0))
        else:
            self.register_parameter('bias', None)
            self.bias = None
        # Adding Parameters and Modules
        super(Linear, self).__init__()
        
    def forward(self, input):
        return F.linear(input, self.weight, self.bias)
    
    def backward(self, output):
        input = output.previous_inputs[0]
        dldx, dldw, dldb = F.d_linear(input, self.weight, output.grad_v)
        self.weight.grad_v += dldw
        if self.bias is not None:
            self.bias.grad_v += dldb
        if isinstance(input, Variable):
            input.grad_v = dldx
            input.backward_run()
        
    def __call__(self, input):
        output = self.forward(input)
        return Variable(output, previous_inputs=(input,) , previous_module=self)
    

class ReLU(Module):
    """
    ReLU activation Layer
    """
    def __init__(self):
        super(ReLU, self).__init__()
        
    def forward(self, input):
        return F.relu(input)
    
    def backward(self, output):
        input = output.previous_inputs[0]
        if isinstance(input, Variable):
            input.grad_v = F.d_relu(input, output.grad_v)
            input.backward_run()
        
    def __call__(self, input):
        output= self.forward(input)
        return Variable(output, previous_inputs=(input,) , previous_module=self)
    
    
class Tanh(Module):
    """
    Tanh Activation Layer
    """
    def __init__(self):
        super(Tanh, self).__init__()
        
    def forward(self, input):
        return F.tanh(input)
    
    def backward(self, output):
        input = output.previous_inputs[0]
        if isinstance(input, Variable):
            input.grad_v = F.d_tanh(input, output.grad_v)
            input.backward_run()
        
    def __call__(self, input):
        output= self.forward(input)
        return Variable(output, previous_inputs=(input,) , previous_module=self)
    

class Softmax(Module):
    """
    Softmax activation Layer
    """
    def __init__(self):
        super(Softmax, self).__init__()
        
    def forward(self, input):
        return F.softmax(input)
    
    def backward(self, output):
        input = output.previous_inputs[0]
        if isinstance(input, Variable):
            input.grad_v = F.d_softmax(input, output.grad_v)
            input.backward_run()
    
    def __call__(self, input):
        output = self.forward(input)
        return Variable(output, previous_inputs=(input,) , previous_module=self)
    
class MSELoss(Module):
    """
    MSE Loss Module
    """
    def __init__(self, reduction='mean'):
        super(MSELoss, self).__init__()
        self.reduction = reduction
    def forward(self, input, target):
        return F.mse_loss(input, target, reduction=self.reduction).view(-1)
    def backward(self, output):
        input = output.previous_inputs[0]
        target = output.previous_inputs[1]
        if isinstance(input, Variable):
            input.grad_v = F.d_mse_loss(input, target)
            input.backward_run()
    def __call__(self, input, target):
        output = self.forward(input, target)
        return Variable(output, previous_inputs=(input,target), previous_module=self)
    
    
class CrossEntropyLoss(Module):
    """
    Cross Entropy Loss
    """
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, input, target):
        return F.cross_entropy(input, target).view(-1)
    
    def backward(self, output):
        input = output.previous_inputs[0]
        target = output.previous_inputs[1]
        if isinstance(input, Variable):
            input.grad_v = F.d_cross_entropy(input, target)
            input.backward_run()
    
    def __call__(self, input, target):
        output = self.forward(input, target)
        return Variable(output, previous_inputs=(input,target), previous_module=self)
    
class Sequential(Module):
    """
    Creates a sequential module contains all sub-modules
    """
    def __init__(self, *args):
        super(Sequential, self).__init__()
        for idx, module in enumerate(args):
            self.register_module(str(idx), module)
            
    def forward(self, input):
        for module in self.modules_.values():
            input = module(input)
        return input
    def __call__(self, input):
        return self.forward(input)
    
    

class Variable(torch.Tensor):
    """
    Variable class: This class is used to store the gradients and all the network graph leafs
    inside it and enables .backward_run() method to run recursively!
    """
    def __new__(cls, x, grad_f=True, grad_v=None, previous_inputs=None, previous_module=None, *args, **kwargs):
        return super().__new__(cls, x, *args, **kwargs)
    
    def __init__(self, x, grad_f=True, grad_v=None, previous_inputs=None, previous_module=None):
        self.grad_f = grad_f
        self.grad_v = grad_v
        self.previous_inputs = previous_inputs
        self.previous_module = previous_module
        
    def backward_run(self):
        if self.grad_v is None:
            self.grad_v = torch.ones(self.shape)
        if self.previous_module is not None:
            self.previous_module.backward(self)
        

class Parameter(torch.Tensor):
    """
    Parameter class: This class is used to define a parameter inside a module. This enables modules to
    automatically see parameters and add them to the module!
    """
    def __new__(cls, x, grad_f=True, grad_v=None, *args, **kwargs):
        return super().__new__(cls, x, *args, **kwargs)
    
    def __init__(self, x, grad_f=True, grad_v=None):
        self.grad_f = grad_f
        self.grad_v = grad_v