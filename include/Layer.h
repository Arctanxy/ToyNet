#pragma once
#include "Matrix.h"
#include "Tensor.h"
#include <map>
#include <string>


class Layer
{
public:
	virtual Tensor forward(Tensor x);
	virtual Tensor backward(Tensor grad); // 需要传入上一层的梯度，并将梯度传入到下一层
};

class Linear :public Layer
{
public:
	int in_dim, out_dim;
	std::map<std::string, Matrix> params;
	Linear(int in_channels, int out_channels);
	Linear() {};
	~Linear() {};
	Tensor forward(Tensor x);
	Tensor backward(Tensor grad);
};

class Conv2d :public Layer
{
public:
	int in_dim, out_dim, k_size, s_size, p_size;
	std::map<std::string, Matrix> params;
	Tensor forward(Tensor x);
	Tensor backward(Tensor grad);
};

class Maxpooling :public Layer
{
public:
	int k_size, s_size, p_size;
	Tensor forward(Tensor x);
	Tensor backward(Tensor grad);
};

class Sigmoid :public Layer
{
public:
	Tensor forward(Tensor x);
	Tensor backward(Tensor grad);
};