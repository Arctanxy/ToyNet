#pragma once
#include <assert.h>
#include "Matrix.h"

enum Op{
    constant, add, minus, multiply, divide, convolution, maxpooling, relu, sigmoid, dot_product
};

class Tensor
{
public:
    // matrix
    Matrix mat;

    // node 
    Tensor *left;
    Tensor *right;
    Op tp;
    bool is_leaf;

    // autograd
    bool requires_grad;
    Matrix grad;
    Matrix *left_grad;
    Matrix *right_grad;
    void backward(Matrix init_grad); // init_grad matrix(1.0,mat.row,mat.col)

    // constructors
    Tensor(Matrix &v, bool req_grad=true);
    Tensor(Tensor &l, Tensor &r, const Op &t);


    // operators
    friend Tensor operator+(Tensor &l,Tensor &r);
    friend Tensor operator-(Tensor &l,Tensor &r);
    friend Tensor operator*(Tensor &l,Tensor &r);
    friend Tensor operator/(Tensor &l,Tensor &r);
    friend std::ostream & operator<<(std::ostream &out, Tensor &t);
	Tensor dot(Tensor &r);
};