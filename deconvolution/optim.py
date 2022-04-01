import numpy as np
from functional import *

class Optimizer:
    def step(self):
        pass

    def n_steps(self, n):
        pass

class GD(Optimizer):
    def __init__(self, func: Func, x0, lr, 
        momentum_factor=0, nesterov=False):
        self.func=func
        self.x=x0
        self.lr=lr
        self.momentum_factor=momentum_factor
        self.momentum=np.zeros(x0.shape)
        self.nesterov=nesterov

    def step(self):
        # self.x=self.x-self.lr*self.func.grad(self.x)
        grad=np.zeros(self.x.shape)
        if self.nesterov:
            grad=self.func.grad(self.x+self.lr*self.momentum_factor*self.momentum)
        else:
            grad=self.func.grad(self.x)
        self.momentum=self.momentum_factor*self.momentum-grad
        self.x=self.x+self.lr*self.momentum
        
    # def n_steps(self, n):
    #     for i in range(n):
    #         self.x=self.x-self.lr*self.func.grad(self.x)

class Scheduler:
    def step(self):
        pass
    # def n_steps(self):
    #     pass

class ReduceLROnPlateau(Scheduler):
    def __init__(self, optimizer: Optimizer, factor=0.1, 
                 patience=10, threshold=1e-4, verbose=False):
        self.optimizer=optimizer
        self.factor=factor
        self.patience=patience
        self.bad_steps=0
        self.best=self.optimizer.func(self.optimizer.x)
        self.threshold=threshold
        self.verbose=verbose

    def step(self):
        if self.bad_steps==self.patience:
            self.optimizer.lr*=self.factor
            self.bad_steps=0
            if self.verbose:
                print(f'lr decay by factor {self.factor}')
        self.optimizer.step()
        if self.optimizer.func(self.optimizer.x)>self.best-self.threshold:
            self.bad_steps+=1
        else:
            self.bad_steps=0 



