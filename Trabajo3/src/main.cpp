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

	cout << endl;
}

/*
Funcion que devuelve un valor aleatorio en el intervalo (0,1]
*/
float generadorAleatorio() {
	return static_cast <float> (rand()+1) / static_cast <float> (RAND_MAX);
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

/*
Generacion de puntos del mundo:
*/
vector<Mat> generarPtosMundo() {
	vector<Mat> puntos_mundo;
	Mat m;

	//Creamos los ptos del mundo como matrices columna con cuatro componente siendo la cuarta siempre 1:
	for (int i = 1; i <= 10;  i++)
		for (int j = 1; j <= 10; j++){
			m = Mat(4, 1, CV_32F, 1.0);
			m.at<float>(0,0) = 0.0; m.at<float>(1,0) = i*0.1; m.at<float>(2,0) = j*0.1;
			puntos_mundo.push_back(m);

			m = Mat(4, 1, CV_32F, 1.0);
			m.at<float>(0,0) = j*0.1; m.at<float>(1,0) = i*0.1; m.at<float>(2,0) = 0.0;
			puntos_mundo.push_back(m);
		}

	/*for (int i = 0; i < puntos_mundo.size(); i++) {
		cout << "Pto" << i << endl;
		mostrarMatriz(puntos_mundo.at(i));
	}*/


	return puntos_mundo;
}


/*
Funcion que obtiene la proyeccion de ptos 3D por medio de una camara.
*/
vector<Mat> obtenerPtosProyectados (vector<Mat> ptos_mundo, Mat camara) {
	vector<Mat> proyecciones;

	//Aplicamos la matriz de camara al vector columna del pto.
	for (int i = 0; i < ptos_mundo.size(); i++)
		proyecciones.push_back(camara * ptos_mundo.at(i));

	//Homogeneizamos la tercera componente de cada pto.
	for (int i = 0; i < proyecciones.size(); i++)
		proyecciones.at(i) = proyecciones.at(i) / proyecciones.at(i).at<float>(2,0);

	return proyecciones;
}


/*
Generacion de la matriz de coeficientes para el algoritmo SLV a partir de ptos del mundo y proyectados.
@ptos_mundos: las coordenadas 3D de los ptos del mundo (matrices 4x1)
@ptos_proyectados: las coordenadas 2D de la proyeccion de los ptos del mundo (matrices 3x1)
*/
Mat obtenerMatrizCoeficientes (vector<Mat> ptos_mundo, vector<Mat> ptos_proyectados) {
	int f = 2 * ptos_mundo.size();
	int c = 12;
	Mat pto_mundo_actual, pto_proyectado_actual;

	//Creamos la matriz de coeficientes inicializada a cero, ahorrando pasos para cuando la instanciemos al caso particular.
	Mat m_coeff = Mat (f, c, CV_32F, 0.0);

	//Rellenamos la matriz segun el esquema que ha detener al resolver el sistema de ecuaciones relacionado con ella.
	for (int i = 0; i < f; i = i+2) {
		pto_mundo_actual = ptos_mundo.at(i/2);
		pto_proyectado_actual = ptos_proyectados.at(i/2);

		m_coeff.at<float>(i,0) = pto_mundo_actual.at<float>(0,0);
		m_coeff.at<float>(i,1) = pto_mundo_actual.at<float>(1,0);
		m_coeff.at<float>(i,2) = pto_mundo_actual.at<float>(2,0);
		m_coeff.at<float>(i,3) = 1.0;

		m_coeff.at<float>(i,8) = -pto_proyectado_actual.at<float>(0,0) * pto_mundo_actual.at<float>(0,0);
		m_coeff.at<float>(i,9) = -pto_proyectado_actual.at<float>(0,0) * pto_mundo_actual.at<float>(1,0);
		m_coeff.at<float>(i,10) = -pto_proyectado_actual.at<float>(0,0) * pto_mundo_actual.at<float>(2,0);
		m_coeff.at<float>(i,11) = -pto_proyectado_actual.at<float>(0,0);

		m_coeff.at<float>(i+1,4) = pto_mundo_actual.at<float>(0,0);
		m_coeff.at<float>(i+1,5) = pto_mundo_actual.at<float>(1,0);
		m_coeff.at<float>(i+1,6) = pto_mundo_actual.at<float>(2,0);
		m_coeff.at<float>(i+1,7) = 1.0;

		m_coeff.at<float>(i+1,8) = -pto_proyectado_actual.at<float>(1,0) * pto_mundo_actual.at<float>(0,0);
		m_coeff.at<float>(i+1,9) = -pto_proyectado_actual.at<float>(1,0) * pto_mundo_actual.at<float>(1,0);
		m_coeff.at<float>(i+1,10) = -pto_proyectado_actual.at<float>(1,0) * pto_mundo_actual.at<float>(2,0);
		m_coeff.at<float>(i+1,11) = -pto_proyectado_actual.at<float>(1,0);

	}


	return m_coeff;
}


/*
Funcion que estima una camara a partir de unos ptos del mundo y de sus proyecciones.
@ptos_mundos: las coordenadas 3D de los ptos del mundo (matrices 4x1)
@ptos_proyectados: las coordenadas 2D de la proyeccion de los ptos del mundo (matrices 3x1)
*/
Mat estimarCamaraDLT (vector<Mat> ptos_mundo, vector<Mat> ptos_proyectados) {
	Mat A, w, u, vt;

	A = obtenerMatrizCoeficientes(ptos_mundo, ptos_proyectados);

	//Obtenemos la descomposicion SVD de la matriz de coeficientes, la matriz que nos interesa es la vT:
	SVD::compute(A, w, u, vt);

	Mat camara_estimada = Mat(3, 4, CV_32F);

	//Construimos la matriz de transformacion con la ultima columna de v o lo que es lo mismo la ultima fila de vT:
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 4; j++)
			camara_estimada.at<float>(i, j) = vt.at<float>(11, i * 4 + j);

	return camara_estimada;
}

/*
Funcion que calcula la distancia entre dos matrices (con las mismas dimensiones) con la norma de Frobenius.
@A: la primera matriz.
@B: la segunda matriz.
*/
double calcularDistanciaMatrices (Mat A, Mat B) {
	//Falta normalizar A y B.
	Mat diferencia = (A/A.at<float>(0,0)) - (B/B.at<float>(0,0));
	int f = diferencia.rows;
	int c = diferencia.cols;

	double sum = 0.0;

	for (int i = 0; i < f; i++)
		for (int j = 0; j < c; j++)
			sum += diferencia.at<float>(i,j) * diferencia.at<float>(i,j);


	return sqrt(sum);
}

/*
Funcion que obtiene las coordenas pixel de un conjunto de puntos 2D en coord. homogeneas y los pinta.
@ptos: conjunto de ptos a dibujar
@im: imagen donde dibujar
*/
void dibujarPtos(vector<Mat> ptos, Mat im, Scalar color) {
	float x_max, x_min, y_max, y_min;
	int f = im.rows; int c = im.cols;

	//Calculamos los rangos maximos de valores donde se mueven las coordenadas de los ptos.
	x_max = ptos.at(0).at<float>(0,0);
	x_min = ptos.at(0).at<float>(0,0);
	y_max = ptos.at(0).at<float>(1,0);
	y_min = ptos.at(0).at<float>(1,0);

	for (int i = 1; i < ptos.size(); i++) {
		if (x_max < ptos.at(i).at<float>(0,0))
			x_max = ptos.at(i).at<float>(0,0);
		else if (x_min > ptos.at(i).at<float>(0,0))
			x_min = ptos.at(i).at<float>(0,0);

		if (y_max < ptos.at(i).at<float>(1,0))
			y_max = ptos.at(i).at<float>(1,0);
		else if (y_min > ptos.at(i).at<float>(1,0))
			y_min = ptos.at(i).at<float>(1,0);
	}

	cout << "El rango maximo de la x es: (" << x_min << "," << x_max << ")" << endl;
	cout << "El rango maximo de la y es: (" << y_min << "," << y_max << ")" << endl;

	float longitud_x = (x_max - x_min);
	float longitud_y = (y_max - y_min);

	cout << "Longitud x: " << longitud_x << endl;
	cout << "Longitud y: " << longitud_y << endl;

	float x, y;
	for (int i = 0; i < ptos.size(); i++) {
		x = ptos.at(i).at<float>(0,0);
		y = ptos.at(i).at<float>(1,0);
		circle(im, Point(ceil((x-x_min)*c*longitud_x),ceil((y-y_min)*f*longitud_y)), 1, color);
	}

}

//Funcion donde se estructuran los pasos necesarios para el primer punto de la practica
void parte1() {
	Mat camara_generada = generarCamaraAleatoria();

	vector<Mat> ptos_mundo = generarPtosMundo();

	vector<Mat> proyecciones_camara_original = obtenerPtosProyectados (ptos_mundo, camara_generada);

	Mat camara_estimada = estimarCamaraDLT(ptos_mundo, proyecciones_camara_original);

	vector<Mat> proyecciones_camara_estimada = obtenerPtosProyectados(ptos_mundo, camara_estimada);

	Mat imagen_ptos_originales = Mat::zeros(500, 500, CV_32FC3);
	Mat imagen_ptos_estimados = Mat::zeros(500, 500, CV_32FC3);

	dibujarPtos(proyecciones_camara_original, imagen_ptos_originales, Scalar(255, 0, 0));
	dibujarPtos(proyecciones_camara_estimada, imagen_ptos_estimados, Scalar(0, 255, 255));

	//imshow("Ptos proyectados (azul) y ptos estimados (rojo)", imagen_ptos);
	imshow("Ptos proyectados", imagen_ptos_originales);
	imshow("Ptos estimados", imagen_ptos_estimados);


	cout << "El error cometido en la aproximacion es: " << calcularDistanciaMatrices(camara_generada, camara_estimada) << endl;

	waitKey(0);
	destroyAllWindows();
}

//Funcion donde se estructuran los pasos necesarios para el segundo punto de la practica.
void parte2() {
	//Cargamos las imagenes en color:
	vector<Mat> imagenes_tablero, imagenes_calibracion;
	vector<Point2f> esquinas_img_actual;
	vector<vector<Point2f>> esquinas_imgs_calibracion;

	tamano_tablero = Size(13, 12);

	for (int i = 1; i <= 25; i++)
		imagenes_tablero.push_back( imread("imagenes/chessboard/Image"+to_string(i)+".tif"));

	//Obtenemos las posiciones de las esquinas del tablero en las imagen donde podamos localizarlas.
	for (int i = 0; i < 25; i++) {
		if ( findChessboardCorners(imagenes_tablero.at(i), tamano_tablero, esquinas_img_actual) ) {
			imagenes_calibracion.push_back(imagenes_tablero.at(i));
			esquinas_imgs_calibracion.push_back(esquinas_img_actual);
		}

		esquinas_img_actual.clear();
	}

	//Refinamos las coordenadas obtenidas anteriormente.
	for (int = 0; imagenes_calibracion.size(); i++) {
		cornerSubPix( imagenes_calibracion.at(i), esquinas_imgs_calibracion.at(i), tamano_tablero, Size(-1,-1), TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ));
	}

	//Pintamos las esquinas encontradas:
	for (int i = 0; imagenes_calibracion.size(); i++) {
		drawChessboardCorners( imagenes_calibracion.at(i), tamano_tablero, Mat(esquinas_imgs_calibracion.at(i)), true);
	}

	vector<Point3f> esquinas_teoricas;

	//Obtenemos los ptos teoricos donde ha de estar el patron que estamos buscando
  for( int i = 0; i < tamano_tablero.height; i++)
    for( int j = 0; j < tamano_tablero.width; j++)
        esquinas_teoricas.push_back(Point3f(float(j), float(i), 0));

	vector<vector<Point3f> > puntos_objeto(1);
	puntos_objeto.resize(imagenes_calibracion.size(), esquinas_teoricas);







}


int main() {
	cout << "*****************************\nPARTE 1: ESTIMACION DE CAMARA\n*****************************" << endl;
	parte1();

	cout << "******************\nPARTE 2: CALIBRACION\n******************" << endl;
	parte2();

}
