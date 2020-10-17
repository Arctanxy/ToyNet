#!/usr/bin/env python
# -*- coding: utf-8 -*-

class Tensor(object):
    def __init__(self, value, parents = None, grad_fns = None, requires_grad = True):
        self.value = value
        self.grad = None 
        self.parents = parents 
        self.grad_fns = grad_fns 
        self.requires_grad = requires_grad

    def __add__(self,t):
        # attrs of result node
        value = self.value + t.value 
        requires_grad = (self.requires_grad or t.requires_grad)
        parents = []
        parents.append(self)
        parents.append(t)
        # backward functions
        grad_fns = []
        grad_fns.append(lambda x: 1 * x)
        grad_fns.append(lambda x: 1 * x)

        return Tensor(value, parents, grad_fns, requires_grad)

    def __sub__(self, t):
        value = self.value - t.value 
        requires_grad = (self.requires_grad or t.requires_grad)
        parents = []
        parents.append(self) 
        parents.append(t)
        # backward functions
        grad_fns = []
        grad_fns.append(lambda x: 1 * x)
        grad_fns.append(lambda x: -1 * x)
        return Tensor(value, parents, grad_fns, requires_grad)

    def __mul__(self, t):
        value = self.value * t.value 
        requires_grad = (self.requires_grad or t.requires_grad)
        parents = []
        parents.append(self) 
        parents.append(t)
        # backward functions
        grad_fns = []
        grad_fns.append(lambda x: t.value * x)
        grad_fns.append(lambda x: self.value * x)
        return Tensor(value, parents, grad_fns, requires_grad)


    def backward(self): 
        # input data or frozen weight doesn't need gradient
        if not self.requires_grad: 
            return 
        # if the grad of this node has not been computed
        # it must be an loss node 
        if self.grad is None:
            self.grad = 1

        # if this node is an operator node, it will have some parents nodes
        if self.parents is not None: 
            for i, p in enumerate(self.parents):
                # if the parent node's grad has not be computed
                if p.grad is None:
                    p.grad = 0
                p.grad += self.grad_fns[i](self.grad)
                p.backward()

    def zero_grad(self):
        if self.parents is not None: 
            for p in self.parents: 
                p.zero_grad()
        self.grad = None 

class Net:
    def __init__(self):
        self.w = Tensor(2, requires_grad=True)
        self.b = Tensor(1, requires_grad = True)
        self.lr = 0.01

    def forward(self,x, y):
        y_hat = self.w * x + self.b 
        z = (y_hat - y) * (y_hat - y)
        return z

    def train_one_epoch(self, x, y):
        z = self.forward(x, y)
        z.backward()
        self.update()
        z.zero_grad()
        return z

    def update(self):
        self.w.value -= self.lr * self.w.grad 
        self.b.value -= self.lr * self.b.grad 




if __name__ == "__main__":
    x = Tensor(1, requires_grad=False)
    y = Tensor(2, requires_grad = False)

    net = Net()
    for i in range(100):
        z = net.train_one_epoch(x, y)
        print(z.value, net.w.value, net.b.value)
