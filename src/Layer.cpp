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

Ŀǰ�����⣺

���Բ�������weight��bias����������������Tensor�����ƣ��Ƿ���Ҫ�޸�Tensor�����ƣ�
���ǽ����Բ���Զ��󵼷ֳɵ�˺ͼӷ�������������󵼣�


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
