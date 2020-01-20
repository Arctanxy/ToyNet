#include "Layer.h"
#include "Matrix.h"
#include <map>
#include <string>

Linear::Linear(int in_channels, int out_channels)
{
	in_dim = in_channels;
	out_dim = out_channels;
	params.insert(std::pair<std::string, Matrix>("weight", Matrix::Matrix(1.0, in_channels, out_channels)));
	params.insert(std::pair<std::string, Matrix>("bias", Matrix::Matrix(1.0, 1, out_channels)));
}


/*

目前的问题：

线性层里面有weight和bias两个参数，超出了Tensor类的设计，是否需要修改Tensor类的设计，
还是将线性层的自动求导分成点乘和加法两个步骤进行求导？


*/

Tensor Linear::forward(Tensor x)
{
	Matrix result = x.mat.dot(params.find("weight")->second) + params.find("bias")->second;
	return result;
}

Tensor Linear::backward(Tensor grad)
{
	grad_w = input.transpose().dot(grad);
	grad_b = grad.sum(0);
	Tensor<float> w_t = weight.transpose();
	return grad.dot(w_t);
}
