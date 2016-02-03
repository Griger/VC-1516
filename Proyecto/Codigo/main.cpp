#include<opencv2/opencv.hpp>
#include<string>
#include<cmath>

using namespace std;
using namespace cv;

void showIm(Mat im) {
	namedWindow("window", 1);
	imshow("window", im);
	//waitKey();
	//destroyWindow("window");
}

Mat curvar_cilindro(Mat im, double f, double s){
	Mat imagen_curvada = Mat::zeros(im.rows, im.cols, CV_8U);

	int centro_x = im.cols/2;
	int centro_y = im.rows/2;

	for(int i = 0; i < im.rows; i++)
		for(int j = 0; j < im.cols; j++)
			imagen_curvada.at<uchar>(floor(s*((i-centro_y)/sqrt((j-centro_x)*(j-centro_x)+f*f)) + centro_y),
								floor(s*atan((j-centro_x)/f) + centro_x) ) = im.at<uchar>(i,j);

	return imagen_curvada;
}

Mat curvar_esfera(Mat im, double f, double s){
	Mat imagen_curvada = Mat::zeros(im.rows, im.cols, CV_8U);

	int centro_x = im.cols/2;
	int centro_y = im.rows/2;

	for(int i = 0; i < im.rows; i++)
		for(int j = 0; j < im.cols; j++)
			imagen_curvada.at<uchar>(floor(s*atan((i-centro_y)/sqrt((j-centro_x)*(j-centro_x)+f*f)) + centro_y),
								floor(s*atan((j-centro_x)/f) + centro_x) ) = im.at<uchar>(i,j);

	return imagen_curvada;
}

int main(int argc, char* argv[]){
	Mat imagen = imread("Image1.tif", 0);

	showIm(imagen);

	Mat imagen_cilindro, imagen_esfera;
	imagen_cilindro = curvar_cilindro(imagen,500,500);
	imagen_esfera = curvar_esfera(imagen,500,500);

	//showIm(imagen_cilindro);
	//showIm(imagen_esfera);
	namedWindow("window1", 1);
	imshow("window1", imagen_cilindro);
	namedWindow("window2", 1);
	imshow("window2", imagen_esfera);
	waitKey();
	destroyWindow("window1");
	destroyWindow("window2");

    return 0;
}
