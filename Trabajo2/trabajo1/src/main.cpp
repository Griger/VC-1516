#include<opencv2/opencv.hpp>
#include<vector>

#include "imagenes.hpp"
#include "convolucion.hpp"

using namespace std;
using namespace cv;

# define M_PI           3.14159265358979323846

void mostrarMatriz(Mat &m) {
	for (int i = 0; i < m.rows; i++) {
		for (int j = 0; j < m.cols; j++)
			cout << "|" << m.at<float>(i, j) << "|";
		cout << endl;
	}

	cout << endl;
}

//BONUS 1:

//Declaramos las funciones que intervienen en las mascaras y de las que por tanto muestrearemos.

float parte1PrimeraDerivada (float x, float sigma) {
	return 1/(2*M_PI*sigma*sigma)*(-x/(sigma*sigma))*exp(-(x*x)/(2*sigma*sigma));
}

float parte2PrimeraDerivada(float x, float sigma) {
	return exp(-(x*x) / (2 * sigma*sigma));
}

float parte1SegundaDerivada(float x, float sigma) {
	return (1 / (2 * M_PI*sigma*sigma)) * (((-1 / (sigma*sigma))*exp(-(x*x) / (2 * sigma*sigma))) + ((-x / (sigma*sigma)) * (-x / (sigma*sigma)) * exp(-(x*x) / (2 * sigma*sigma))));
}

float parte2SegundaDerivada(float x, float sigma) {
	return exp(-(x*x) / (2 * sigma*sigma));
}

/*
Funcion que calcula las dos mascaras en las que se puede descomponer la mascara 2D de la primera parcial (con respecto a x o y) de una Gaussiana.
@sigma: el parametro sigma del que dependen las funciones de la mascara.
@parte1: matriz donde se almacenara la primera parte de la mascara (la relacionada con la funcion parte1PrimeraDerivada).
@parte2: matriz donde se almacenara la segunda parte de la mascara (la relacionada con la funcion parte2PrimeraDerivada).
*/
void calcularMascarasPrimeraDerivada(float sigma, Mat &parte1, Mat &parte2) {
	parte1 = calcularVectorMascara(sigma, parte1PrimeraDerivada);
	parte2 = calcularVectorMascara(sigma, parte2PrimeraDerivada);
}

/*
Funcion que calcula las dos mascaras en las que se puede descomponer la mascara 2D de la segunda parcial (con respecto a x o y, dos veces) de una Gaussiana.
@sigma: el parametro sigma del que dependen las funciones de la mascara.
@parte1: matriz donde se almacenara la primera parte de la mascara (la relacionada con la funcion parte1SegundaDerivada).
@parte2: matriz donde se almacenara la segunda parte de la mascara (la relacionada con la funcion parte2SegundaDerivada).
*/
void calcularMascarasSegundaDerivada(float sigma, Mat &parte1, Mat &parte2) {
	parte1 = calcularVectorMascara(sigma, parte1SegundaDerivada);
	parte2 = calcularVectorMascara(sigma, parte2SegundaDerivada);
}

class Punto{
public:
	float x, y;

	Punto() {
		x = 0.0;
		y = 0.0;
	}

	Punto(float a, float b) {
		x = a;
		y = b;
	}
};

/*
Funcion que obtiene la matriz de coeficientes para el sistema del calculo de la homografia dados los puntos muestreados en las im�genes estudiadas.
@puntos_origen: puntos muestreados en la imagen de partida.
@puntos_destino: puntos muestreados en la imagen de destino.
*/
Mat obtenerMatrizCoeficientes(vector<Punto> puntos_origen, vector<Punto> puntos_destino) {
	int puntos_muestreados = puntos_origen.size();
	Mat A = Mat(2 * puntos_muestreados, 9, CV_32F, 0.0);
	Punto punto_og, punto_dst;

	//Construimos la matriz de coeficientes A tal y como lo tenemos en las trasparencias.
	for (int i = 0; i < 2*puntos_muestreados; i = i+2) {
		punto_og = puntos_origen.at(i/2);
		punto_dst = puntos_destino.at(i/2);
		A.at<float>(i, 0) = punto_og.x; A.at<float>(i, 1) = punto_og.y; A.at<float>(i, 2) = 1.0;
		A.at<float>(i, 6) = -punto_dst.x*punto_og.x; A.at<float>(i, 7) = -punto_dst.x*punto_og.y; A.at<float>(i, 8) = -punto_dst.x;
		A.at<float>(i+1, 3) = punto_og.x; A.at<float>(i+1, 4) = punto_og.y; A.at<float>(i+1, 5) = 1.0;
		A.at<float>(i+1, 6) = -punto_dst.y*punto_og.x; A.at<float>(i+1, 7) = -punto_dst.y*punto_og.y; A.at<float>(i+1, 8) = -punto_dst.y;
	}

	return A;
}

/*
Funci�n que obtiene la matriz de trasnformacion que lleva una imagen a la otra, de forma aproximada:
@puntos_origen: puntos muestreados en la imagen de partida.
@puntos_destino: puntos muestreados en la imagen de destino.
*/
Mat obtenerMatrizTransformacion(vector<Punto> puntos_origen, vector<Punto> puntos_destino) {
	Mat A, w, u, vt;

	//obtenemos la matriz de coeficientes:
	A = obtenerMatrizCoeficientes(puntos_origen, puntos_destino);

	//Obtenemos la descomposicion SVD de la matriz de coeficientes, la matriz que nos interesa es la vT:
	SVD::compute(A, w, u, vt);

	Mat H = Mat(3, 3, CV_32F);

	//Construimos la matriz de transformacion con la ultima columna de v o lo que es lo mismo la ultima fila de vT:
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			H.at<float>(i, j) = vt.at<float>(8, i * 3 + j);

	return H;
}

/*
Funcion que obtiene los KeyPoints de una imagen con el detector BRISK.
@im: imagen a la que le calculamos los KeyPoints
@umbral: parametro de umbral (thresh) para el detector BRISK a usar.
@octavas: parametro de octavas (octaves) para el detector BRISK a usar.
@escalaDePatron: parametro de escala del patron (patternScale) para el detector BRISK a usar.
*/

vector<KeyPoint> obtenerKeyPointsBRISK (Mat im, int umbral = 30, int octavas = 3, float escalaDePatron = 1.0f) {
	Ptr<BRISK> ptrDetectorBRISK = BRISK::create(umbral, octavas, escalaDePAtron);
	vector<KeyPoint> puntosDetectados;

	ptrDetectorBRISK->detect(im, puntosDetectados);

	cout << "Hemos obtenido: " << puntosDetectados.size() << " puntos." << endl;

	return puntos;
}

/*
Funcion que obtiene los KeyPoints de una imagen con el detector ORB.
@im: imagen a la que le calculamos los KeyPoints
*/
vector<KeyPoint> obtenerKeyPointsORB (Mat im, int num_caracteristicas = 500, float factor_escala = 1.2f, int num_niveles = 8, int umbral_borde = 31, int WTA_K = 2, int tamano_area = 31, int umbral_rapido = 20, int tipo_marcador = ORB::HARRIS_SCORE){
	Ptr<ORB> prtDetectorORB = ORB::create(num_caracteristicas, factor_escala, num_niveles, umbral_borde, 0, WTA_K, tipo_marcador, tamano_area, umbral_rapido);
	vector<KeyPoint> puntosDetectados;

	ptrDetectorORB->detect(im, puntosDetectados);

	cout << "Hemos obtenido: " << puntosDetectados.size() << " puntos." << endl;

}



int main(int argc, char* argv[]) {

	cout << "OpenCV detectada " << endl;

/*
=====================================
PARTE 1: ESTIMACION DE LA HOMOGRAFIA
=====================================
*/

	/*Mat tablero1 = imread("imagenes/Tablero1.jpg");
	Mat tablero2 = imread("imagenes/Tablero2.jpg");

	vector<Punto> ptos_tablero1;
	vector<Punto> ptos_tablero2;

	//Almacenamos las coordenadas de los ptos en correspondencia que hemos muestreado de cada imagen:
	ptos_tablero1.push_back(Punto(175, 70));
	ptos_tablero1.push_back(Punto(532, 41));
	ptos_tablero1.push_back(Punto(158, 423));
	ptos_tablero1.push_back(Punto(526, 464));
	ptos_tablero1.push_back(Punto(261, 165));
	ptos_tablero1.push_back(Punto(416, 160));
	ptos_tablero1.push_back(Punto(254, 327));
	ptos_tablero1.push_back(Punto(413, 335));
	ptos_tablero1.push_back(Punto(311, 109));
	ptos_tablero1.push_back(Punto(355, 390));

	ptos_tablero2.push_back(Punto(168, 46));
	ptos_tablero2.push_back(Punto(500, 120));
	ptos_tablero2.push_back(Punto(101, 392));
	ptos_tablero2.push_back(Punto(432, 442));
	ptos_tablero2.push_back(Punto(248, 164));
	ptos_tablero2.push_back(Punto(391, 195));
	ptos_tablero2.push_back(Punto(218, 312));
	ptos_tablero2.push_back(Punto(362, 337));
	ptos_tablero2.push_back(Punto(306, 125));
	ptos_tablero2.push_back(Punto(305, 377));

	Mat H = obtenerMatrizTransformacion(ptos_tablero1, ptos_tablero2);
	Mat tablero1_transformada;

	warpPerspective(tablero1, tablero1_transformada, H, Size(tablero2.cols, tablero2.rows));

	vector<Mat> imagenes;

	imagenes.push_back(tablero1);
	imagenes.push_back(tablero1_transformada);
	imagenes.push_back(tablero2);


	mostrarImagenes("Calcular homografia (Aparado 1):", imagenes);*/

/*
==========================
PARTE 2: EXTRAER KEYPOINTS
==========================
*/

	//Cargamos las imagenes para los apartados 2 y 3:
	Mat yose1 = imread("imagenes/Yosemite1.jpg");
	Mat yose2 = imread("imagenes/Yosemite2.jpg");

	//Las mostramos para ver que se han cargado correctamente (GUSI quitar):
	imshow("Yose1", yose1);
	imshow("Yose2", yose2);

	vector<KeyPoint> puntosDetectados; //donde almacenaremos los puntos detectados para cada imagen por cada criterio, reutilizable.
	Mat yose1KPBRISK, yose1KPORB, yose2KPBRISK, yose2KPBRISK; //las imagenes correspondientes pintando los puntos detectados.

	puntosDetectados = obtenerKeyPointsBRISK(yose1);
	drawKeyPoints(yose1, puntosDetectados, yose1KPBRISK);
	puntosDetectados = obtenerKeyPointsORB(yose1);
	drawKeyPoints(yose1, puntosDetectados, yose1KPORB);
	puntosDetectados = obtenerKeyPointsBRISK(yose2);
	drawKeyPoints(yose2, puntosDetectados, yose2KPBRISK);
	puntosDetectados = obtenerKeyPointsORB(yose2);
	drawKeyPoints(yose2, puntosDetectados, yose2KPORB);





	waitKey(0);
	destroyAllWindows();

	return 0;
}
