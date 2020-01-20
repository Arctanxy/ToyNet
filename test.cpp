#include "Tensor.h"
#include "Matrix.h"
#include <iostream>


int main()
{
	Matrix m(5.0, 1, 1);
	Matrix n(3.0, 1, 1);
	Tensor a(m), b(n);
	Tensor c = a - b * a;
    std::cout << a.grad << b.grad << std::endl;
	Matrix init_grad(1.0, c.mat.row, c.mat.col);
	c.backward(init_grad);
	std::cout << a.grad << b.grad << std::endl;
	getchar();
    return 1;
}