# Deep Learning Mini Projects
This repositories contains the project files for the "Deep Learning 2019" mini projects.

# General Decription
The codes have been implemented in [Python 3.6.x](https://www.python.org/downloads/release/python-360/) using [PyTorch 1.0.x](https://pytorch.org/get-started/locally/) framework. If you are using newer version of PyTorch, please check for the compatibility of the codes.

## Mini Project 1: Different Arcitechtures in Image Classification
Image classification is a very common task in the field of Computer Vision. In this project the objective is to see how different architectures behave in comparing two digits represented as 14x14 images.
Several techniques can be used to carry out this task and among these. We have tried three different approaches to solve this problem, including direct comparison with/without auxiliary loss, and direct digit classification using a combination of fully connected and convolutional layers.  
More information can be found in the report: [Mini Project 1: Description](https://github.com/rezaho/deep_learning_mini_projects/blob/master/Proj1/report.pdf) .  
The source codes can be found in the `./Proj1/` directory. For testing the project run `./Proj1/test.py` source file.

## Mini Project 2: Designing a deep learning framework
In this project, we have tried to design our mini deep learning framework from the scratch.
As for the purpose of this project, we only used basic `torch.Tensor` object without any Autograd functions or Neural Network modules. For this purpose we turned off the Autograd machinery using `torch.set_grad_enabled(False)`.
The logic behind the codes was inspired from PyTorch library, and then we added our ideas to make it fit to the purpose of this project.  
More information can be found in the report: [Mini Prject 2: Description](https://github.com/rezaho/deep_learning_mini_projects/blob/master/Proj2/report.pdf).  
The source codes can be found in the `./Proj2/` directory. For testing the project, run `./Proj2/text.py` source file.


## Aknowledgements
This project was done in collaboration with [@alex-mocanu](https://github.com/alex-mocanu/) and [@mammadhajili](https://github.com/mammadhajili/).
