#ifndef CONVOLUCION_HPP
#define CONVOLUCION_HPP

#include<opencv2/opencv.hpp>

using namespace cv;
using namespace std;

float f(float x, float sigma);
Mat calcularVectorMascara(float sigma, float(*f)(float, float));
Mat obtenerVectorOrlado1C(Mat &senal, Mat &mascara, int cond_contorno);
Mat calcularConvolucionVectores1C(Mat &senal, Mat &mascara, int cond_contorno);
Mat convolucion2D1C(Mat &im, float sigma, int cond_bordes);
Mat convolucion2D(Mat &im, float sigma, int cond_bordes);



#endif