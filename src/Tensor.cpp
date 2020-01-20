#include "Tensor.h"
#include "Matrix.h"


// Constructors
// leaf node
Tensor::Tensor(Matrix &m,bool req_grad)
{
    mat = m;
    left = NULL;
    right = NULL;
    is_leaf = true;
    requires_grad = req_grad;
	grad = Matrix::Matrix(0.0, m.row, m.col);
}

Tensor::Tensor(Tensor &l, Tensor &r, const Op &t)
{
    left = &l;
    right = &r;
	tp = t;
    if(l.requires_grad || r.requires_grad)
    {
        requires_grad = true;
		grad = Matrix::Matrix(0.0, l.mat.row, l.mat.col);
    }
}

// Ops
Tensor operator+(Tensor &l, Tensor &r)
{
    Tensor out(l,r,add);
    Matrix m = l.mat + r.mat;
    out.mat = m;
	return out;
}

Tensor operator-(Tensor &l, Tensor &r)
{
    Tensor out(l,r,minus);
    Matrix m = l.mat - r.mat;
    out.mat = m;
	return out;
}

Tensor operator*(Tensor &l, Tensor &r)
{
    Tensor out(l,r,multiply);
    Matrix m = l.mat * r.mat;
    out.mat = m;
	return out;
}

Tensor operator/(Tensor &l, Tensor &r)
{
    Tensor out(l,r,divide);
    Matrix m = l.mat / r.mat;
    out.mat = m;
	return out;
}

Tensor Tensor::dot(Tensor &r)
{
	Tensor out(*this, r, dot_product);
	Matrix m = mat.dot(r.mat);
	out.mat = m;
	return out;
}

// Backward
void Tensor::backward(Matrix init_grad)
{
    if(left)
    {
		if (tp == add)
		{
			left->grad = left->grad + 1.0 * init_grad;
		}
		
		
		
		switch (tp)
        {
		case constant:
			break;
        case add:
            left->grad = left->grad + 1.0 * init_grad;
            break;
        case minus:
			left->grad = left->grad + 1.0 * init_grad;
            break;
        case multiply:
			left->grad = left->grad + right->mat * init_grad;
            break;
        case divide:
			left->grad = left->grad + (1.0 / right->mat) * init_grad;
            break;
		case convolution:
			break;
		case maxpooling:
			break;
		case relu:
			break;
		case sigmoid:
			break;
        default:
            break;
        }
		left->backward(left->grad);
    }
	if (right)
	{
		switch (tp)
		{
		case constant:
			break;
		case add:
			right->grad = right->grad + 1.0 * init_grad;
			break;
		case minus:
			right->grad = right->grad - 1.0 * init_grad;
			break;
		case multiply:
			right->grad = right->grad + left->mat * init_grad;
			break;
		case divide:
			right->grad = right->grad - left->mat * (1 / (right->mat * right->mat)) * init_grad;
			break;
		case convolution:
			break;
		case maxpooling:
			break;
		case relu:
			break;
		case sigmoid:
			break;
		default:
			break;
		}
		right->backward(right->grad);
	}
}