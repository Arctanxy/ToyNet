import matplotlib.pyplot as plt
from matplotlib import animation




class Tensor:
    def __init__(self,data,left=None,right=None,op = None):
        self.data = data
        self.grad = 0
        self.left = left
        self.right = right
        self.op = op

    def __add__(self, other):
        data = self.data + other.data
        t = Tensor(data,left = self,right=other,op = "add")
        return t

    def __sub__(self, other):
        data = self.data - other.data
        t = Tensor(data,left = self,right=other,op = "sub")
        return t

    def __mul__(self, other):
        data = self.data * other.data
        t = Tensor(data, left=self, right=other, op="mul")
        return t

    def __truediv__(self, other):
        if other.data - 0 < 1e-9:
            raise Exception("Can't divide zero")
        data = self.data / other.data
        t = Tensor(data, left=self, right=other, op="div")
        return t

    def backward(self,init_grad = 1):
        # init_grad： 来自上一层的梯度
        if self.left is not None:
            if self.op == "add":
                self.left.grad += 1 * init_grad
            elif self.op == "sub":
                self.left.grad += 1 * init_grad
            elif self.op == "mul":
                self.left.grad += self.right.data * init_grad
            elif self.op == "div":
                self.left.grad += 1 / self.right.data * init_grad
            else:
                raise Exception("Op unacceptable")
            self.left.backward(self.left.grad)
        if self.right is not None:
            if self.op == "add":
                self.right.grad += 1 * init_grad
            elif self.op == "sub":
                self.right.grad += -1 * init_grad
            elif self.op == "mul":
                self.right.grad += self.left.data * init_grad
            elif self.op == "div":
                self.right.grad += (-1 * self.left.data / (self.right.data*self.right.data)) * init_grad
            else:
                raise Exception("Op unacceptable")
            self.right.backward(self.right.grad)

class Linear_regression:
    def __init__(self):
        self.w = Tensor(1.0)
        self.b = Tensor(1.0)
        self.lr = Tensor(0.02)

    def fit(self,x,y,num_epochs = 60,show=True):
        if show:
            fig = plt.figure()
            plt.scatter(x,y,color = 'r')
            ims = []

        for epoch in range(num_epochs):
            losses = 0.0
            for m,n in zip(x,y):
                yp = self.w * Tensor(m) + self.b
                loss = (Tensor(n) - yp) * (Tensor(n) - yp)
                loss.backward()
                self.w -= self.lr * Tensor(self.w.grad)
                self.b -= self.lr * Tensor(self.b.grad)
                self.w.grad = 0
                self.b.grad = 0
                # 切断计算图
                self.w.left = None
                self.w.right = None
                self.b.right = None
                self.b.left = None
                losses += loss.data
            print(losses)

            if show:
                im = plt.plot(x,[self.w.data * item + self.b.data for item in x],color = 'g')
                ims.append(im)
            if show:
                ani = animation.ArtistAnimation(fig, ims, interval=200,
                                            repeat_delay=1000)
                ani.save("test.gif", writer='pillow')


if __name__ == "__main__":
    TEST = False
    TRAIN = True
    if TEST:
        a = Tensor(1.0)
        b = Tensor(2.0)
        c = a * b + a / b - a * a * a
        c.backward()
        print("grad \na:{} b:{}".format(a.grad,b.grad))
        import torch
        # 需要用小写的torch.tensor才能添加requires_grad参数
        m = torch.tensor([[1.0]],requires_grad=True)
        n = torch.tensor([[2.0]],requires_grad=True)
        k = m * n + m / n - m * m * m
        k.backward()
        print("grad torch\nm:{} n:{}".format(m.grad.item(),n.grad.item()))
    if TRAIN:
        x = [1,2,3,4,5]
        y = [6,5,4,3,2]
        clf = Linear_regression()
        clf.fit(x,y)
