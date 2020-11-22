#include <string>


class Mat
{
public:
    Mat(const Mat& mat); // copy construction
    // Mat(const std::string &path); // load image
    Mat(const int channel, const int width, const int height);
    Mat(const int channel, const int width, const int height, void *data);
    Mat();
    ~Mat();

    // void readImage(const std::string &path);
    // void saveImage(const std::string &path);

    void release();

    int getChannel();
    int getWidth();
    int getHeight();

    void * getData();

    bool isEmpty();
    int getSize();

    friend Mat operator+(const Mat &A, const Mat &B);
    friend Mat operator+(const double &a, const Mat &B);
    friend Mat operator+(const Mat &A, const double &b);

    friend Mat operator-(const Mat &A, const Mat &B);
    friend Mat operator-(const double &a, const Mat &B);
    friend Mat operator-(const Mat &A, const double &b);
    
    friend Mat operator*(const Mat &A, const Mat &B);
    friend Mat operator*(const double &a, const Mat &B);
    friend Mat operator*(const Mat &A, const double &b);

    friend Mat operator/(const Mat &A, const Mat &B);
    friend Mat operator/(const double &a, const Mat &B);
    friend Mat operator/(const Mat &A, const double &b);


    protected:
    int _channel = 0;
    int _width = 0;
    int _height = 0;
    void * data;
};