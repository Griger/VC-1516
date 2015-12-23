#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdlib.h>

using namespace std;
using namespace cv;

void mostrarMatriz(Mat m) {
	int f = m.rows;
	int c = m.cols;
	
	for (int i = 0; i < f; i++){
		for (int j = 0; j < c; j++)
			cout << m.at<float>(i,j) << " ";
		cout << endl;
	}
}

/*
Funcion que devuelve un valor aleatorio en el intervalo [0,1]
*/
float generadorAleatorio() {
	return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
}

/*
Funcion que genera una camara aleatoria 
*/
Mat generarCamaraAleatoria() {
	Mat camara = Mat(3,4,CV_32F);
	bool camara_correcta = false;

	//Mientras que la camara no tenga el menor distinto de cero
	while (!camara_correcta) {
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 4; j++)
				camara.at<float>(i,j) = generadorAleatorio();

		//Tomamos la submatriz a la que queremos calcular el determinante
		Mat submatriz = Mat(camara, Range::all(), Range(0,3));
		
		if (determinant(submatriz) != 0)
			camara_correcta = true;
	}

	return camara;
}

//Generacion de puntos del mundo:
vector<Point3f> generarPtosMundo() {
	vector<Point3f> puntos_mundo;
	for (int i = 1; i <= 10;  i++)
		for (int j = 1; j <= 10; j++){
			puntos_mundo.push_back(Point3f(0,i*0.1, j*0.1));
			puntos_mundo.push_back(Point3f(j*0.1, i*0.1, 0));
		}

	return puntos_mundo;
}

//Funcion donde se estructura los pasos necesarios para el primer punto de la practica
void parte1() {
	Mat camara_generada = generarCamaraAleatoria();
	
	cout << "La camara generada es:" << endl;
	mostrarMatriz(camara_generada);
	
	
	//ptos_mundo = generarPtosMundo();
	
	


}


int main() {
	cout << "*****************************\nPARTE 1: ESTIMACION DE CAMARA\n*****************************" << endl;
	parte1();
}
