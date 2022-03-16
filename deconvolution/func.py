import numpy as np
import types


class Func:
    def __call__(self, x):
        pass

    def grad(self):
        pass
    
    def __add__(self, other):
        if type(other) in (int, float):
            FuncSum=types.new_class('FuncSum', bases=(Func,))
            FuncSum.__call__=lambda obj, x: self(x)+other
            FuncSum.grad=lambda obj: self.grad()

        elif issubclass(type(other), Func):
            FuncSum=types.new_class('FuncSum', bases=(Func,))
            FuncSum.__call__=lambda obj, x: self(x)+other(x)
            FuncSum.grad=lambda obj: self.grad()+other.grad()

        else:
            raise ValueError

        return FuncSum()
    
    def __mul__(self, other):
        if type(other) in (int, float):
            FuncMul=types.new_class('FuncMul', bases=(Func,))
            FuncMul.__call__=lambda obj, x: self(x)*other
            FuncMul.grad=lambda obj: self.grad()*other

        elif issubclass(type(other), Func):
            FuncMul=types.new_class('FuncMul', bases=(Func,))
            FuncMul.__call__=lambda obj, x: self(x)*other(x)
            FuncMul.grad=lambda obj: (self.grad()*other +
                self*other.grad())

        else:
            raise ValueError
        
        return FuncMul()

    __radd__=__add__
    __rmul__=__mul__

def grad(func: Func):
    return func.grad()
    

class Exp(Func):
    def __call__(self, x):
        return np.exp(x)

    def grad(self):
        return Exp()

class Sin(Func):
    def __call__(self, x):
        return np.sin(x)

    def grad(self):
        return Cos()

class Cos(Func):
    def __call__(self, x):
        return np.cos(x)

    def grad(self):
        return -1*Sin()



def main():
    sin=Sin()
    exp=Exp()

    print((3+sin)(0))
    print(grad(3+sin)(0), '\n')

    print((sin*3)(np.pi/2))
    print(grad(3*sin)(0), '\n')

    print((sin+exp)(0))
    print((sin+exp).grad()(0), '\n')
    
    print((exp*sin+12)(np.pi/2))
    print(grad(3*exp*sin+12)(np.pi/2)-3*np.exp(np.pi/2)*np.sin(np.pi/2))



if __name__=='__main__':
    main()