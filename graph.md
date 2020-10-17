# 神经网络框架中的动态图与静态图

在讨论神经网络训练框架的时候，总会提到动态计算图与静态计算图。

静态图需要先构建再运行，优势是在运行前可以对图结构进行优化，比如常数折叠、算子融合等，可以获得更快的前向运算速度。

缺点也很明显，就是只有在计算图运行起来之后，才能看到变量的值，像TensorFlow1.x中的session.run那样。

动态图是一边运行一边构建，优势是可以在搭建网络的时候看见变量的值，便于检查。

缺点是前向运算不好优化，因为根本不知道下一步运算要算什么。

但是我在用过PyTorch和TensorFlow1.x之后，并没有感受到这种理论上的前向运算速度差距，只感受到了动态图的便利。


所以从TensorFlow2.x将Eager模式设置成默认模式之后，除PyTorch之外，其他的热门框架都已经有了静态图和动态图两套方案了。

两种计算图方案的实现方式略有不同，本文将用Python演示如何实现动态图与静态图。

为了偷懒: 

* 算子只实现+-×

* 使用标量运算 

## 动态图

动态图的实现较为简单，因为只有在反向传播的过程中才会实际用到这个图结构，所以在设计数据结构的时候，只需要记录父节点即可。

```python
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

if __name__ == "__main__":
    x = Tensor(1, requires_grad=False)
    y = Tensor(2, requires_grad = False)
    w = Tensor(2, requires_grad = True)
    b = Tensor(1, requires_grad = True)

    lr = 0.01
    for i in range(100):
        # forward and backward
        y_hat = w * x + b 
        z = (y_hat - y) * (y_hat - y)
        z.backward()
        
        # update weights
        w.value -= lr * w.grad 
        b.value -= lr * b.grad 

        print(z.value, w.value, b.value, w.grad, b.grad)
        # clear gradient
        z.zero_grad()
```
## 静态图

相比之下，静态图的定义更抽象一些，为了更好地认识静态图的运算过程，我们可以将Graph类单独提取出来。

```python 
class Graph:
    def __init__(self):
        self.nodes = []

    def add_nodes(self,node):
        self.nodes.append(node)

    def zero_grad(self):
        for node in self.nodes:
            node.zero_grad()
    
    def backward(self): 
        for node in self.nodes: 
            node.backward()


default_graph = Graph()

class Node:
    def __init__(self, *parents):
        self.value = None
        self.grad = None 
        self.parents = parents 
        self.children = []
        self.graph = default_graph 
        self.is_end_node = False  # endnode is usually a loss node 
        self.waiting_for_forward = True # forward function has not been called 
        self.waiting_for_backward = True # backward function has not been called 

        # add current node to the children list of parents 
        for p in self.parents:
            p.children.append(self)

        # add current node to graph 
        self.graph.add_nodes(self)

    def forward(self): 
        for p in self.parents:
            if p.waiting_for_forward:
                p.forward()
        self.forward_single()

    def backward(self):
        for c in self.children: 
            if c.waiting_for_backward:
                c.backward()
        self.backward_single()

    def forward_single(self):
        pass

    def backward_single(self):
        pass 

    def zero_grad(self):
        self.grad = None
        self.waiting_for_backward = True
        self.waiting_for_forward = True

class Add(Node):
    def forward_single(self):
        assert(len(self.parents) == 2)
        # ignore if this node is not waiting for forward
        if not self.waiting_for_forward:
            return
        self.value = self.parents[0].value + self.parents[1].value 
        self.waiting_for_forward = False

    def backward_single(self):
        # ignore if this node is not waiting for backward 
        if not self.waiting_for_backward:
            return
        if self.is_end_node:
            self.grad = 1
        for p in self.parents: 
            if p.grad is None:
                p.grad = 0

        self.parents[0].grad += self.grad * 1
        self.parents[1].grad += self.grad * 1
        self.waiting_for_backward = False

class Sub(Node): 
    def forward_single(self):
        assert(len(self.parents) == 2)
        if not self.waiting_for_forward: 
            return

        self.value = self.parents[0].value - self.parents[1].value 
        self.waiting_for_forward = False

    def backward_single(self):
        if not self.waiting_for_backward: 
            return
        if self.is_end_node:
            self.grad = 1
        for p in self.parents:
            if p.grad is None: 
                p.grad = 0

        self.parents[0].grad += self.grad * 1
        self.parents[1].grad += self.grad * (-1)
        self.waiting_for_backward = False

class Mul(Node):
    def forward_single(self):
        assert(len(self.parents) == 2)
        if not self.waiting_for_forward: 
            return
        
        self.value = self.parents[0].value * self.parents[1].value 
        self.waiting_for_forward = False

    def backward_single(self):
        if not self.waiting_for_backward:
            return

        if self.is_end_node:
            self.grad = 1
        for p in self.parents:
            if p.grad is None:
                p.grad = 0

        self.parents[0].grad += self.grad * self.parents[1].value 
        self.parents[1].grad += self.grad * self.parents[0].value 
        self.waiting_for_backward = False 

class Variable(Node):
    def __init__(self, data, requires_grad = True):
        Node.__init__(self)
        self.value = data 
        self.requires_grad = requires_grad 

class SGD(object): 
    def __init__(self, graph, target_node, learning_rate):
        self.graph = graph 
        self.lr = learning_rate 
        self.target = target_node
        self.target.is_end_node = True

    def zero_grad(self):
        # clear the gradient in graph
        self.graph.zero_grad()

    def get_grad(self):
        # get gradient all over the graph
        self.target.forward()
        self.graph.backward()
    
    def step(self):
        # update weights
        for node in self.graph.nodes:
            if not (isinstance(node, Variable) and node.requires_grad == False):
                node.value -= self.lr * node.grad 

if __name__ == "__main__":
    x = Variable(1, False)
    y = Variable(2, False)
    w = Variable(2, True)
    b = Variable(3, True)

    y_hat = Add(Mul(w, x), b) # y_hat = w * x + b
    loss = Mul(Sub(y_hat, y), Sub(y_hat, y)) # loss = (y_hat - y) * (y_hat - y)

    optimizer = SGD(default_graph, loss, 0.01)
    for i in range(100): 
        optimizer.zero_grad()
        optimizer.get_grad()
        optimizer.step()
        print(loss.value,w.grad,b.grad)

```

完整代码参见：https://github.com/Arctanxy/ToyNet


PyTorch动态图的backward逻辑参考：[autograd的python接口](https://github.com/pytorch/pytorch/blob/415ed434aaaffb7dd89bbce9c8db8beaa4562483/torch/csrc/autograd/python_engine.cpp)

关于Node和Edge的介绍在[csrc/autograd/function.h](https://github.com/pytorch/pytorch/blob/415ed434aa/torch/csrc/autograd/function.h)
