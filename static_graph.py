
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

