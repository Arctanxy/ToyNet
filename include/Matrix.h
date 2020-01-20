#pragma once
#include <ostream>

class Matrix
{
public:
    // attr
    int row,col;
    float *data;

    // Ops
    friend Matrix operator+(Matrix &m, Matrix &n);
    friend Matrix operator-(Matrix &m, Matrix &n);
    friend Matrix operator*(Matrix &m, Matrix &n);
    friend Matrix operator*(float a, Matrix &m);
    friend Matrix operator/(Matrix &m, Matrix &n);
    friend Matrix operator/(float a, Matrix &m);
    void operator=(Matrix &m)
    {
        data = m.data;
        row = m.row;
        col = m.col;
    }
    friend std::ostream & operator<<(std::ostream &out, Matrix &m);
    Matrix transpose();
    Matrix sum(int dim = 0);
    Matrix dot(const Matrix &m);
    Matrix clip(int start, int end);
    Matrix clip(int limit, bool start);

    // constructors
    Matrix(){}; // default
    Matrix(int r, int c);
    Matrix(float v, int r, int c); // constant matrix
    Matrix(float *v, int r, int c);
};