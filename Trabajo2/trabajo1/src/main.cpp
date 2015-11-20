#include<opencv2/opencv.hpp>
#include<vector>
#include<string>

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

vector<KeyPoint> obtenerKeyPointsBRISK (Mat im, int umbral = 30) {
	Ptr<BRISK> ptrDetectorBRISK = BRISK::create(umbral);
	vector<KeyPoint> puntosDetectados;

	ptrDetectorBRISK->detect(im, puntosDetectados);

	cout << "Hemos obtenido: " << puntosDetectados.size() << " puntos." << endl;

	return puntosDetectados;
}

/*
Funcion que obtiene los descriptores de los KeyPoints localizados mediante un detector BRISK
@im: imagen a la que le buscamos los descriptores
@umbral: parametro de umbral para el detector BRISK a usar
*/
Mat obtenerDescriptoresBRISK (Mat im, int umbral = 30) {
	Ptr<BRISK> ptrDetectorBRISK = BRISK::create(umbral);
	vector<KeyPoint> puntosDetectados;
	Mat descriptores;
	
	ptrDetectorBRISK->detect(im, puntosDetectados);
	
	ptrDetectorBRISK->compute(im, puntosDetectados, descriptores);
	
	return descriptores;
}

/*
Funcion que calcula los puntos en correspondencias entre dos imagen por el criterio de Fuerza Bruta + comprobacion cruzada
@im1 e im2: las imagenes entre las cuales vamos a buscar puntos en correspondencias.
@umbral: el umbral para el detector BRISK.
*/
vector<DMatch> obtenerMatchesFuerzaBruta (Mat im1, Mat im2, int umbral){
	vector<KeyPoint> puntosDetectados1, puntosDetectados2;
	Mat descriptores1, descriptores2, imagenMatches;
	vector<DMatch> matches;
	
	//Creamos el matcher con Fuerza Bruta activandole el flag para el cross check.
	BFMatcher matcher = BFMatcher(NORM_L2, true);
	
	puntosDetectados1 = obtenerKeyPointsBRISK(im1, umbral);
	puntosDetectados2 = obtenerKeyPointsBRISK(im2, umbral);
	
	
	//Obtenemos los descriptores de los puntos obtenidos en cada imagen.
	descriptores1 = obtenerDescriptoresBRISK(im1, umbral);
	descriptores2 = obtenerDescriptoresBRISK(im2, umbral);
	
	matcher.match(descriptores1, descriptores2, matches);
	
	drawMatches( im1, puntosDetectados1, im2, puntosDetectados2, matches, imagenMatches, Scalar::all(-1), Scalar::all(-1),                vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
	
	imshow("Matches Fuerza Bruta", imagenMatches);	
	
	return matches;
}

/*
Funcion que calcula los puntos en correspondencias entre dos imagen por el criterio Flann
@im1 e im2: las imagenes entre las cuales vamos a buscar puntos en correspondencias.
@umbral: el umbral para el detector BRISK.
*/
vector<DMatch> obtenerMatchesFlann (Mat im1, Mat im2, int umbral){
	vector<KeyPoint> puntosDetectados1, puntosDetectados2;
	Mat descriptores1, descriptores2, imagenMatches;
	vector<DMatch> matches;
	
	//Creamos el matcher con Fuerza Bruta activandole el flag para el cross check.
	FlannBasedMatcher matcher;
	
	puntosDetectados1 = obtenerKeyPointsBRISK(im1, umbral);
	puntosDetectados2 = obtenerKeyPointsBRISK(im2, umbral);
	
	
	//Obtenemos los descriptores de los puntos obtenidos en cada imagen.
	descriptores1 = obtenerDescriptoresBRISK(im1, umbral);
	descriptores2 = obtenerDescriptoresBRISK(im2, umbral);
	
	//Convertimos las matrices de descriptores a CV_32F para el correcto funcionamiento del matcher creado:
	descriptores1.convertTo(descriptores1, CV_32F);
	descriptores2.convertTo(descriptores2, CV_32F);
	
	
	matcher.match(descriptores1, descriptores2, matches);
	
	drawMatches( im1, puntosDetectados1, im2, puntosDetectados2, matches, imagenMatches, Scalar::all(-1), Scalar::all(-1),                vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
	
	imshow("Matches Flann", imagenMatches);
	
	
	
	return matches;
}




/*
Funcion que obtiene los KeyPoints de una imagen con el detector ORB.
@im: imagen a la que le calculamos los KeyPoints
@num_caracteristicas: numero maximo de caracteristicas a detectar
@tipo_marcador: criterio para elegir o no un punto HARRIS o FAST
@umbral_FAST: umbral para elegir los puntos segun la medidad elegida
*/
vector<KeyPoint> obtenerKeyPointsORB (Mat im, int num_caracteristicas = 500, int tipo_marcador = ORB::HARRIS_SCORE, int umbral_FAST = 20){
	Ptr<ORB> ptrDetectorORB = ORB::create(num_caracteristicas, 1.2f, 8, 31, 0, 2, tipo_marcador, 31, umbral_FAST);
	vector<KeyPoint> puntosDetectados;

	ptrDetectorORB->detect(im, puntosDetectados);

	cout << "Hemos obtenido: " << puntosDetectados.size() << " puntos." << endl;
	
	return puntosDetectados;

}

/*
Funcion que calcula la homografia que lleva la imagen origen a la imagen destino por medio de findHomography
@origen: imagen de origen
@destino: imagen de destino
*/
Mat calcularHomografia (Mat origen, Mat destino) {
	vector<KeyPoint> puntosDetectadosOrigen, puntosDetectadosDestino;
	vector<DMatch> matches;
	vector<Point2f> puntosEnCorrespondenciasOrigen, puntosEnCorrespondenciasDestino;
	
	//Obtenemos los puntos clave con BRISK en cada imagen
	puntosDetectadosOrigen = obtenerKeyPointsBRISK(origen, 65);
	puntosDetectadosDestino = obtenerKeyPointsBRISK(destino, 65);
	
	//Obtenemos los matches por fuerza bruta:
	matches = obtenerMatchesFuerzaBruta(origen, destino, 65);
	
	//Obtenemos los puntos en correspondencias entre ambas imagenes:
	for (int i = 0; i < matches.size(); i++){
		puntosEnCorrespondenciasOrigen.push_back(puntosDetectadosOrigen[matches[i].queryIdx].pt);
		puntosEnCorrespondenciasDestino.push_back(puntosDetectadosDestino[matches[i].trainIdx].pt);	
	}
	
	//Calculamos la homografia con los puntos en correspondencias:
	
	return findHomography(puntosEnCorrespondenciasOrigen, puntosEnCorrespondenciasDestino, CV_RANSAC);
}


/*
Funcion que obtiene un mosaico de proyeccion plana de dos imagenes
@origen y destino: imagenes con las que formar el mosaico.
*/

void mosaicoDeDos (Mat origen, Mat destino) {
	Mat mosaico = Mat(1000, 1000, origen.type());
		
	//Colocamos la primera imagen en la esquina superior izquierda por medio de la identidad:
	Mat id = Mat(3,3,CV_32F,0.0);
	
	for (int i = 0; i < 3; i++)
		id.at<float>(i,i) = 1.0;
		
	warpPerspective(destino, mosaico, id, Size(mosaico.cols, mosaico.rows), INTER_LINEAR, BORDER_CONSTANT);
	
	Mat homografia = calcularHomografia(origen, destino);
	
	warpPerspective(origen, mosaico, homografia, Size(mosaico.cols, mosaico.rows), INTER_LINEAR, BORDER_TRANSPARENT);
	
	imshow("Mosaico de 2", mosaico);
}

/*
Funcion que obtiene un mosaico de proyeccion plana con varias imagen
@imagenes: imagenes para construir el mosaico
*/

void mosaicoDeN (vector<Mat> imagenes) {
	Mat mosaico = Mat(1000, 1000, imagenes.at(0).type());
	int posicion_central = imagenes.size()/2 + 1;
	
	//Colocamos la imagen central del vector en el centro del mosaico
	Mat colocacionCentral = Mat(3,3,CV_32F,0,0);
	
	for (int i = 0; i < 3; i++)
		colocacionCentral.at<float>(i,i) = 1.0;
		
	//Realizamos una traslacion simplemente:
	colocacionCentral.at<float>(0,2) = mosaico.cols/2 - imagenes.at(posicion_central).cols;
	colocacionCentral.at<float>(1,2) = mosaico.rows/2 - imagenes.at(posicion_central).rows;
	
	warpPerspective(imagenes.at(posicion_central), mosaico, colocacionCentral, Size(mosaico.cols, mosaico.rows), INTER_LINEAR, BORDER_CONSTANT);
	
	
	
	imshow("Mosaico de N", mosaico);




}




int main(int argc, char* argv[]) {

	cout << "OpenCV detectada " << endl;

/*
=====================================
PARTE 1: ESTIMACION DE LA HOMOGRAFIA
=====================================
*/
	/*
	Mat tablero1 = imread("imagenes/Tablero1.jpg");
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

	//Aquí hacemos lo mismo sólo que ahora lo hacemos con los puntos para el experimento con puntos no adecuados:
	ptos_tablero1.push_back(Punto(157, 47));
	ptos_tablero1.push_back(Punto(177, 47));
	ptos_tablero1.push_back(Punto(155, 73));
	ptos_tablero1.push_back(Punto(175, 71));
	ptos_tablero1.push_back(Punto(199, 44));
	ptos_tablero1.push_back(Punto(223, 43));
	ptos_tablero1.push_back(Punto(197, 68));
	ptos_tablero1.push_back(Punto(153, 94));
	ptos_tablero1.push_back(Punto(174, 95));
	ptos_tablero1.push_back(Punto(153, 119));

	ptos_tablero2.push_back(Punto(149, 14));
	ptos_tablero2.push_back(Punto(174, 20));
	ptos_tablero2.push_back(Punto(142, 40));
	ptos_tablero2.push_back(Punto(167, 46));
	ptos_tablero2.push_back(Punto(198, 25));
	ptos_tablero2.push_back(Punto(223, 30));
	ptos_tablero2.push_back(Punto(191, 51));
	ptos_tablero2.push_back(Punto(137, 64));
	ptos_tablero2.push_back(Punto(163, 69));
	ptos_tablero2.push_back(Punto(131, 90));

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

	vector<KeyPoint> puntosDetectados; //donde almacenaremos los puntos detectados para cada imagen por cada criterio, reutilizable.
	Mat yose1KPBRISK, yose1KPORB, yose2KPBRISK, yose2KPORB; //las imagenes correspondientes pintando los puntos detectados.

	/*puntosDetectados = obtenerKeyPointsBRISK(yose1, 65);
	drawKeypoints(yose1, puntosDetectados, yose1KPBRISK);
	imshow("Yose 1 KP BRISK", yose1KPBRISK);
	puntosDetectados = obtenerKeyPointsBRISK(yose2, 65);
	drawKeypoints(yose2, puntosDetectados, yose2KPBRISK);
	imshow("Yose 2 KP BRISK", yose2KPBRISK);
	puntosDetectados = obtenerKeyPointsORB(yose1, 1000, ORB::HARRIS_SCORE, 35);
	drawKeypoints(yose1, puntosDetectados, yose1KPORB);
	imshow("Yose 1 KP ORB", yose1KPORB);	
	puntosDetectados = obtenerKeyPointsORB(yose2, 1000, ORB::HARRIS_SCORE, 35);
	drawKeypoints(yose2, puntosDetectados, yose2KPORB);
	imshow("Yose 2 KP ORB", yose2KPORB);*/


/*
===============================
PARTE 3: DESCRIPTORES Y MATCHES
===============================
*/

	//obtenerMatchesFuerzaBruta (yose1, yose2, 65);
	//obtenerMatchesFlann (yose1, yose2, 65);


/*
=================================
PARTE 4: MOSAICO CON DOS IMAGENES
=================================
*/

	//mosaicoDeDos(yose2, yose1);
	
/*
====================================
PARTE 5: MOSAICO CON VARIAS IMAGENES
====================================
*/
	vector<Mat> imagenes;
	
	imagenes.push_back(imread("imagenes/mosaico002.jpg"));
	imagenes.push_back(imread("imagenes/mosaico003.jpg"));
	imagenes.push_back(imread("imagenes/mosaico004.jpg"));
	imagenes.push_back(imread("imagenes/mosaico005.jpg"));
	imagenes.push_back(imread("imagenes/mosaico006.jpg"));
	imagenes.push_back(imread("imagenes/mosaico007.jpg"));
	imagenes.push_back(imread("imagenes/mosaico008.jpg"));
	imagenes.push_back(imread("imagenes/mosaico009.jpg"));
	imagenes.push_back(imread("imagenes/mosaico010.jpg"));
	imagenes.push_back(imread("imagenes/mosaico011.jpg"));
	
	for (int i = 0; i < imagenes.size(); i++){
		string nombreBase = "Mosaico ";
		string numero = to_string(i);
		imshow(nombreBase + numero, imagenes.at(i));
	}
		
	waitKey(0);
	destroyAllWindows();

	return 0;
}
