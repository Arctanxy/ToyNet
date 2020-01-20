#include "Matrix.h"
#include <malloc.h>
#include <assert.h>

// Ops
Matrix operator+(Matrix &m, Matrix &n)
{
    assert(m.row == n.row && m.col == n.col);
    Matrix out(m.row,m.col); // malloc
    for(int i =0;i<m.row;i++)
    {
        for(int j = 0;j<m.col;j++)
        {
            out.data[i*m.col + j] = m.data[i*m.col + j] + n.data[i*m.col + j];
        }
    }
    return out;
}

Matrix operator-(Matrix &m, Matrix &n)
{
    assert(m.row == n.row && m.col == n.col);
    Matrix out(m.row,m.col); // malloc
    for(int i =0;i<m.row;i++)
    {
        for(int j = 0;j<m.col;j++)
        {
            out.data[i*m.col + j] = m.data[i*m.col + j] - n.data[i*m.col + j];
        }
    }
    return out;
}

Matrix operator*(Matrix &m, Matrix &n)
{
    assert(m.row == n.row && m.col == n.col);
    Matrix out(m.row,m.col); // malloc
    for(int i =0;i<m.row;i++)
    {
        for(int j = 0;j<m.col;j++)
        {
            out.data[i*m.col + j] = m.data[i*m.col + j] * n.data[i*m.col + j];
        }
    }
    return out;
}

Matrix operator*(float a, Matrix &m)
{
    Matrix out(m.row,m.col);
    for(int i =0;i<m.row;i++)
    {
        for (int j = 0;j<m.col;j++)
        {
            out.data[i*m.col +j] = m.data[i*m.col + j] * a;
        }
    }
    return out;
}

Matrix operator/(Matrix &m, Matrix &n)
{
    assert(m.row == n.row && m.col == n.col);
    Matrix out(m.row,m.col); // malloc
    for(int i =0;i<m.row;i++)
    {
        for(int j = 0;j<m.col;j++)
        {
            assert(n.data[i*m.col + j] - 0.0 > 1e-9);
            out.data[i*m.col + j] = m.data[i*m.col + j] / n.data[i*m.col + j];
        }
    }
    return out;
}


Matrix operator/(float a, Matrix &m)
{
    Matrix out(m.row,m.col); // malloc
    for(int i =0;i<m.row;i++)
    {
        for(int j = 0;j<m.col;j++)
        {
            assert(m.data[i*m.col + j] - 0.0 > 1e-9);
            out.data[i*m.col + j] = a / m.data[i*m.col + j];
        }
    }
    return out;
}


std::ostream & operator<<(std::ostream &out, Matrix &m)
{
    out << "[";
    for(int i=0;i<m.row;i++)
    {
        out << "[";
        for(int j =0;j<m.col;j++)
        {
            out << m.data[i*m.col +j];
            if(j != m.col - 1)
            {
                out << "\t";
            }
        }
        out << "]";
        if(i != m.row - 1)
        {
            out << "\n";
        }
    }
    out << "]" << "\n";
    return out;
}

Matrix Matrix::transpose()
{
    Matrix out(col,row);
    for (int i =0;i<row;i++)
    {
        for (int j = 0;j<col;j++)
        {
            out.data[j*row + i] = data[i*col +j];
        }
    }
    return out;
}

Matrix Matrix::sum(int dim)
{
    if(dim == 0){
        Matrix out(row,1);
        for(int i = 0;i<row;i++)
        {
            for (int j =0;j<col;j++)
            {
                out.data[i] += data[i*col + j];
            }
        }
        return out;
    }else{
        Matrix out(1,col);
        for(int i = 0;i<row;i++)
        {
            for (int j =0;j<col;j++)
            {
                out.data[j] += data[i*col +j];
            }
        }
        return out;
    }
}

Matrix Matrix::dot(const Matrix &m)
{
    assert(col == m.row);
    Matrix out(row,m.col);
    // out[i,j] = sum(data[i,k] * m[k,j]) for k in range(col)
    for(int i =0;i<row;i++)
    {
        for(int j = 0;j<m.col;j++)
        {
            for(int k = 0;k<col;k++)
            {
                out.data[i*m.col +j] += data[i*col +k] * m.data[k*m.col +j];
            }
        }
    }
    return out;
}


Matrix Matrix::clip(int start, int end)
{
    Matrix out(row,col);
    for(int i =0;i<row;i++)
    {
        for(int j =0;j<col;j++)
        {
            if(out.data[i*col +j] < start)
            {
                out.data[i*col +j] = start;
            }else if(out.data[i*col +j] > end)
            {
                out.data[i*col +j] = end;
            }
        }
    }
    return out;
}

Matrix Matrix::clip(int limit, bool start)
{
    Matrix out(row,col);
    for(int i =0;i<row;i++)
    {
        for (int j =0;j<col;j++)
        {
            if(start)
            {
                if(out.data[i*col +j] < limit)
                {
                    out.data[i*col +j] = limit;
                }
            }else{
                if(out.data[i*col +j] > limit)
                {
                    out.data[i*col +j] = limit;
                }
            }
        }
    }
    return out;
}


// Constructors

Matrix::Matrix(int r, int c)
{
    row = r;
    col = c;
    data = (float*)malloc(r * c * sizeof(float));
}

Matrix::Matrix(float v, int r, int c)
{
    row = r;
    col = c;
    data = (float*)malloc(r * c * sizeof(float));
    for(int i = 0;i<row;i++)
    {
        for(int j = 0;j<col;j++)
        {
            data[i*col +j] = v;
        }
    }
}

Matrix::Matrix(float *v, int r, int c)
{
    row = r;
    col = c;
    data = (float*)malloc(r * c * sizeof(float));
    for(int i = 0;i<row;i++)
    {
        for(int j = 0;j<col;j++)
        {
            data[i*col +j] = v[i*col +j];
        }
    }
}

