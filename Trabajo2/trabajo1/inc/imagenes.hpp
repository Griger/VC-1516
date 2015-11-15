#ifndef IMAGENES_HPP
#define IMAGENES_HPP

#include<opencv2/opencv.hpp>

using namespace cv;
using namespace std;

Mat leeImagen(string nombreArchivo, int flagColor = 1);
float cambioDeRango(float t, float a, float b, float c, float d);
Mat reajustarRango1C(Mat im);
Mat reajustarRango(Mat im);
void mostrarImagen(string nombreVentana, Mat &im, int tipoVentana = 1);
void mostrarImagenes(string nombreVentana, vector<Mat> &imagenes);
void modificarPixeles(Mat &im, vector<Point> &coordenadas, int color1 = 0, int color2 = 0, int color3 = 0);




#endif