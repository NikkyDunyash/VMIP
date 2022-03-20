import types
from matplotlib.pyplot import axis
import numpy as np
import scipy.ndimage

# Simplified func: grad returns value

class Func:
    def _getval(x):
        pass

    def __call__(self, arg):
        if issubclass(type(arg), Func):
            FuncCompose=types.new_class('FuncSum', bases=(Func,))
            FuncCompose.__call__=lambda obj, x: self(arg(x))
            FuncCompose.grad=lambda obj, x: self.grad(arg(x))*arg.grad(x)
            return FuncCompose()
        else:
            return self._getval(arg)     

    def grad(self, x):
        pass
    
    def __add__(self, other):
        if type(other) in (int, float):
            FuncSum=types.new_class('FuncSum', bases=(Func,))
            FuncSum.__call__=lambda obj, x: self(x)+other
            FuncSum.grad=lambda obj, x: self.grad(x)

        elif issubclass(type(other), Func):
            FuncSum=types.new_class('FuncSum', bases=(Func,))
            FuncSum.__call__=lambda obj, x: self(x)+other(x)
            FuncSum.grad=lambda obj, x: self.grad(x)+other.grad(x)

        else:
            raise ValueError

        return FuncSum()
    
    def __sub__(self, other):
        return self+(-1)*other
    
    def __rsub__(self, other):
        return other+(-1)*self
    
    def __mul__(self, other):
        if type(other) in (int, float):
            FuncMul=types.new_class('FuncMul', bases=(Func,))
            FuncMul.__call__=lambda obj, x: self(x)*other
            FuncMul.grad=lambda obj, x: self.grad(x)*other

        elif issubclass(type(other), Func):
            FuncMul=types.new_class('FuncMul', bases=(Func,))
            FuncMul.__call__=lambda obj, x: self(x)*other(x)
            FuncMul.grad=lambda obj, x: (self.grad(x)*other(x) +
                self(x)*other.grad(x))

        else:
            raise ValueError
        
        return FuncMul()

    __radd__=__add__
    __rmul__=__mul__



def grad(func: Func, x):
    return func.grad(x)
    
class Power(Func):
    def __init__(self, p):
        self.p=p

    def _getval(self, x):  
        x=np.array(x)
        return np.power(x, self.p)
    
    def grad(self, x):
        return self.p*np.power(x, self.p-1)

class Exp(Func):
    def _getval(self, x):
        return np.exp(x)

    def grad(self, x):
        return np.exp(x)

class Sin(Func):
    def _getval(self, x):
        return np.sin(x)

    def grad(self, x):
        return np.cos(x)

class Cos(Func):
    def _getval(self, x):
        return np.cos(x)

    def grad(self, x):
        return -np.sin(x)
    

# class LinearTr(Func):
#     pass

class Id(Func):
    def _getval(self, x):
        return x

    def T(self):
        return Id()

class Conv(Func):
    def __init__(self, kernel):
        self.kernel=np.array(kernel)

    def _getval(self, x):
        return scipy.ndimage.convolve(x, weights=self.kernel, mode='nearest')

    def T(self):
        return Conv(np.flip(self.kernel))
    

class ResL1(Func):
    def __init__(self, A=Id(), b=0):
        self.A=A
        self.b=np.array(b)
    
    def _getval(self, x):
        return np.sum(np.abs(self.A(x)-self.b))
    
    def grad(self, x):
        return self.A.T()(np.sign(self.A(x)-self.b))

class ResL2(Func):
    def __init__(self, A=Id(), b=0):
        self.A=A
        self.b=np.array(b)
    
    def _getval(self, x):
        return np.sum((self.A(x)-self.b)**2)
    
    def grad(self, x):
        return 2*self.A.T()(self.A(x)-self.b)
    


class Shift(Func):
    def __init__(self,  axis=0, shift=1):
        if axis<0:
            raise ValueError
        self.axis=axis
        self.shift=shift
        
    def _getval(self, x):
        x=np.array(x)
        pad_width=tuple([(np.abs(self.shift), np.abs(self.shift))
            if i==self.axis else (0, 0) for i in range(x.ndim)])
        shifted_x=np.pad(x, pad_width, mode='edge')
        if self.shift>=0:
            shifted_x=np.take(shifted_x, 
                indices=range(2*self.shift, x.shape[self.axis]+2*self.shift),
                axis=self.axis)
        else:
            shifted_x=np.take(shifted_x, 
                indices=range(x.shape[self.axis]),
                axis=self.axis)
        return shifted_x

    def T(self): 
        return Shift(axis=self.axis, shift=-self.shift)


class TV(Func):
    def __init__(self, directions=[[0 , 1], [1, 0]]):
        self.directions=np.array(directions)

    def _getval(self, x):
        result=0
        x=np.array(x)
        shift_x=Shift()
        shift_y=Shift(axis=1)
        for direction in self.directions:
            shift_x.shift=direction[0]
            shift_y.shift=direction[1]
            result+=(np.sum(np.abs(shift_x(shift_y(x))-x))
                /(np.sum(direction**2)**(0.5)))
        return result

    def grad(self, x):
        result=0
        x=np.array(x)
        shift_x=Shift()
        shift_y=Shift(axis=1)
        for direction in self.directions:
            shift_x.shift=direction[0]
            shift_y.shift=direction[1]
            res_sign=np.sign(shift_x(shift_y(x))-x)
            result+=( (shift_x.T()(shift_y.T())-Id())(res_sign)
                /(np.sum(direction**2)**(0.5)))
        return result


class TV2(Func):
    def __init__(self, directions=[[0 , 1], [1, 0]]):
        self.directions=np.array(directions)
    
    def _getval(self, x):
        result=0
        x=np.array(x)
        shift_x=Shift()
        shift_y=Shift(axis=1)
        for direction in self.directions:
            shift_x.shift=direction[0]
            shift_y.shift=direction[1]
            result+=(np.sum(np.abs(shift_x(shift_y(x))
                +shift_x.T()(shift_y.T()(x))-2*x))
                /(np.sum(direction**2)**(0.5)))
        return result

    def grad(self, x):
        result=0
        x=np.array(x)
        shift_x=Shift()
        shift_y=Shift(axis=1)
        for direction in self.directions:
            shift_x.shift=direction[0]
            shift_y.shift=direction[1]
            res_sign=np.sign(shift_x(shift_y(x))
                +shift_x.T()(shift_y.T()(x))-2*x)
            result+=((shift_x(shift_y)+shift_x.T()(shift_y.T())-2*Id())(res_sign)
                /(np.sum(direction**2)**(0.5)))
        return result


            

def main():
    sin=Sin()
    cos=Cos()
    exp=Exp()

    # print((sin*3-1)(np.pi/2))
    # print((3*sin+2).grad(0), '\n')

    # print((sin+exp)(0))
    # print((sin+exp).grad(0), '\n')
    
    # print((exp*sin+12)(np.pi/2))
    # print(grad(3*exp*sin+12, np.pi/2)-3*np.exp(np.pi/2)*np.sin(np.pi/2))

    # print(exp(sin)(np.pi/2))
    # print(exp(sin).grad(np.pi/2))
    # print(exp(sin(cos))(np.pi/2))
    # print(exp(cos(cos)).grad(1)
    #     -exp(cos(cos(1)))*sin(cos(1))*sin(1), '\n')

    # res_l1=ResL1(A=Conv(np.ones((3,3))), b=np.eye(3))
    # print(res_l1.grad(np.ones((3,3))))

    # print(Shift(shift=-1, axis=1)([[1, 2, 3], [3, 4, 5]]))

    # print(TV(directions=[[0, 1], [1, 0], [0, -1], [-1, 0]])
    #     .grad([[1, 1, 1], [1, 0, 1], [1, 1, 1]]))


if __name__=='__main__':
    main()