#include <opencv2/opencv.hpp>
#include <vector>
#include<time.h>

using namespace std;
using namespace cv;

/*
Funcion que muestra una imagen por pantalla.
@ nombreVentana: el nombre de la ventana que crearemos y en la que mostraremos la imagen.
@ im: la imagen a mostrar
@ tipoVentana: el tipo de ventana que queremos crear, por defecto es 1, es decir una ventana que se ajusta al tamanio de la imagen.
*/
void mostrarImagen(string nombreVentana, Mat &im, int tipoVentana) {
	if (!im.empty()) {
		namedWindow(nombreVentana, tipoVentana);
		imshow(nombreVentana, im);
	}
	else
		cout << "La imagen no se cargo correctamente" << endl;
}


/*
Funcion que combina varias imagenes en una sola.
@nombreVentana = nombre de la ventana en la que mostrar la imagen resultante.
@imagenes = lista de imagenes a mostrar.
*/
void mostrarImagenes(string nombreVentana, vector<Mat> &imagenes) {
	int colCollage = 0, filCollage = 0; //aqui almacenaremos las columnas y las filas de la imagen que sera la union de todas las demas.
	int contadorColumnas = 0;
	for (Mat & im : imagenes) {
		if (im.channels() < 3) cvtColor(im, im, CV_GRAY2RGB); //cambiamos la codificacion del color de algunas im?genes para que no nos de fallos al crear el collage.
		colCollage += im.cols; //sumamos las columnas
		if (im.rows > filCollage) filCollage = im.rows; //obtenemos el maximo numero de filas que vamos a necesitar
	}

	Mat collage = Mat::zeros(filCollage, colCollage, CV_8UC3); //Creamos la imagen con las dimensiones deseadas y todos los p?xeles inicializados a cero.

	Rect roi; //objeto con el que definiremos el ROI
	Mat imroi; //submatriz de la imagen Collage donde copiaremos la imagen que corresponda.

	for (Mat & im : imagenes) {
		roi = Rect(contadorColumnas, 0, im.cols, im.rows); //Vamos moviendo el ROI para que las imagenes queden juntas y ajustandolo al tamaño de la imagen que corresponda.
		contadorColumnas += im.cols;
		imroi = Mat(collage, roi); //(stackoverflow.com/questions/8206466/how-to-set-roi-in-opencv)

		im.copyTo(imroi);
		//Lo que sucede cuando no coincide el formato de destino y el de origen es que lo que se copia son pixeles negros.
	}

	mostrarImagen(nombreVentana, collage, 0);

}


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
Funcion que obtiene la matriz de coeficientes para el sistema del calculo de la homografia dados los puntos muestreados en las imagenes estudiadas.
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
Funcion que obtiene la matriz de trasnformacion que lleva una imagen a la otra, de forma aproximada:
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
*/

vector<KeyPoint> obtenerKeyPointsBRISK (Mat im, int umbral = 30) {
	//Creamos el detector
	Ptr<BRISK> ptrDetectorBRISK = BRISK::create(umbral);
	vector<KeyPoint> puntosDetectados;
	
	//Obtenemos los KP:
	ptrDetectorBRISK->detect(im, puntosDetectados);

	return puntosDetectados;
}

/*
Funcion que obtiene los KeyPoints de una imagen con el detector ORB.
@im: imagen a la que le calculamos los KeyPoints
@num_caracteristicas: numero maximo de caracteristicas a detectar
@tipo_marcador: criterio para elegir o no un punto HARRIS o FAST
@umbral_FAST: umbral para elegir los puntos segun la medidad elegida
*/
vector<KeyPoint> obtenerKeyPointsORB (Mat im, int num_caracteristicas = 500, int tipo_marcador = ORB::HARRIS_SCORE, int umbral_FAST = 20){
	//Creamos el detector:
	Ptr<ORB> ptrDetectorORB = ORB::create(num_caracteristicas, 1.2f, 8, 31, 0, 2, tipo_marcador, 31, umbral_FAST);
	vector<KeyPoint> puntosDetectados;
	
	//Obtenemos los KP:
	ptrDetectorORB->detect(im, puntosDetectados);

	return puntosDetectados;
}


/*
Funcion que obtiene los descriptores de los KeyPoints localizados mediante un detector BRISK
@im: imagen a la que le buscamos los descriptores
@umbral: parametro de umbral para el detector BRISK a usar
*/
Mat obtenerDescriptoresBRISK (Mat im, int umbral = 30) {
	//Creamos el detector:
	Ptr<BRISK> ptrDetectorBRISK = BRISK::create(umbral);
	vector<KeyPoint> puntosDetectados;
	Mat descriptores;
	
	//Obtenemos los KP:
	ptrDetectorBRISK->detect(im, puntosDetectados);
	
	//Obtenemos los descriptores para estos KP:
	ptrDetectorBRISK->compute(im, puntosDetectados, descriptores);
	
	return descriptores;
}

/*
Funcion que obtiene los descriptores de los KeyPoints localizados mediante un detector ORB
@im: imagen a la que le buscamos los descriptores
@num_caracteristicas: numero maximo de caracteristicas a detectar
@tipo_marcador: criterio para elegir o no un punto HARRIS o FAST
@umbral_FAST: umbral para elegir los puntos segun la medidad elegida
*/
Mat obtenerDescriptoresORB (Mat im, int num_caracteristicas = 500, int tipo_marcador = ORB::HARRIS_SCORE, int umbral_FAST = 20) {
	//Creamos el detector:
	Ptr<ORB> ptrDetectorORB = ORB::create(num_caracteristicas, 1.2f, 8, 31, 0, 2, tipo_marcador, 31, umbral_FAST);
	vector<KeyPoint> puntosDetectados;
	Mat descriptores;
	
	//Obtenemos los KP:
	ptrDetectorORB->detect(im, puntosDetectados);
	
	//Obtenemos los descriptores para estos KP:
	ptrDetectorORB->compute(im, puntosDetectados, descriptores);
	
	return descriptores;
}


/*
Funcion que calcula los puntos en correspondencias entre dos imagen por el criterio de Fuerza Bruta + comprobacion cruzada + BRISK
@im1 e im2: las imagenes entre las cuales vamos a buscar puntos en correspondencias.
@umbral: el umbral para el detector BRISK.
*/
vector<DMatch> obtenerMatchesFuerzaBrutaBRISK (Mat im1, Mat im2, int umbral){
	vector<KeyPoint> puntosDetectados1, puntosDetectados2;
	Mat descriptores1, descriptores2;
	vector<DMatch> matches;
	
	//Creamos el matcher con Fuerza Bruta activandole el flag para el cross check.
	BFMatcher matcher = BFMatcher(NORM_L2, true);
	
	//Obtenemos los Key Points con BRISK:
	puntosDetectados1 = obtenerKeyPointsBRISK(im1, umbral);
	puntosDetectados2 = obtenerKeyPointsBRISK(im2, umbral);
	
	
	//Obtenemos los descriptores de los puntos obtenidos en cada imagen.
	descriptores1 = obtenerDescriptoresBRISK(im1, umbral);
	descriptores2 = obtenerDescriptoresBRISK(im2, umbral);
	
	//clock_t t_inicio= clock();
	//Calculamos los matches entre ambas imagenes:
	matcher.match(descriptores1, descriptores2, matches);		
	//printf("FB ha tardado: %.2fs\n",(double)(clock() - t_inicio)/CLOCKS_PER_SEC);
	
	return matches;
}

/*
Funcion que calcula los puntos en correspondencias entre dos imagen por el criterio de Fuerza Bruta + comprobacion cruzada + ORB
@im1 e im2: las imagenes entre las cuales vamos a buscar puntos en correspondencias.
*/
vector<DMatch> obtenerMatchesFuerzaBrutaORB (Mat im1, Mat im2){
	vector<KeyPoint> puntosDetectados1, puntosDetectados2;
	Mat descriptores1, descriptores2;
	vector<DMatch> matches;
	
	//Creamos el matcher con Fuerza Bruta activandole el flag para el cross check.
	BFMatcher matcher = BFMatcher(NORM_L2, true);
	
	//Obtenemos los Key Points con ORB:
	puntosDetectados1 = obtenerKeyPointsORB(im1, 1000, ORB::HARRIS_SCORE, 35);
	puntosDetectados2 = obtenerKeyPointsORB(im2, 1000, ORB::HARRIS_SCORE, 35);
	
	
	//Obtenemos los descriptores de los puntos obtenidos en cada imagen.
	descriptores1 = obtenerDescriptoresORB(im1, 1000, ORB::HARRIS_SCORE, 35);
	descriptores2 = obtenerDescriptoresORB(im2, 1000, ORB::HARRIS_SCORE, 35);
	
	//Calculamos los matches entre ambas imagenes:
	matcher.match(descriptores1, descriptores2, matches);		
	
	return matches;
}


/*
Funcion que calcula los puntos en correspondencias entre dos imagen por el criterio Flann + BRISK
@im1 e im2: las imagenes entre las cuales vamos a buscar puntos en correspondencias.
@umbral: el umbral para el detector BRISK.
*/
vector<DMatch> obtenerMatchesFlannBRISK (Mat im1, Mat im2, int umbral){
	vector<KeyPoint> puntosDetectados1, puntosDetectados2;
	Mat descriptores1, descriptores2;
	vector<DMatch> matches;
	
	//Creamos el matcher con FLANN.
	FlannBasedMatcher matcher;
	
	//Obtenemos los Key Points con BRISK:
	puntosDetectados1 = obtenerKeyPointsBRISK(im1, umbral);
	puntosDetectados2 = obtenerKeyPointsBRISK(im2, umbral);
	
	
	//Obtenemos los descriptores de los puntos obtenidos en cada imagen.
	descriptores1 = obtenerDescriptoresBRISK(im1, umbral);
	descriptores2 = obtenerDescriptoresBRISK(im2, umbral);
	
	//Convertimos las matrices de descriptores a CV_32F para el correcto funcionamiento del matcher creado:
	descriptores1.convertTo(descriptores1, CV_32F);
	descriptores2.convertTo(descriptores2, CV_32F);
	
	//clock_t t_inicio = clock();
	matcher.match(descriptores1, descriptores2, matches);
	//printf("FLANN ha tardado: %.2fs\n",(double)(clock() - t_inicio)/CLOCKS_PER_SEC);
		
	
	return matches;
}

/*
Funcion que calcula los puntos en correspondencias entre dos imagen por el criterio Flann + ORB
@im1 e im2: las imagenes entre las cuales vamos a buscar puntos en correspondencias.
*/
vector<DMatch> obtenerMatchesFlannORB (Mat im1, Mat im2){
	vector<KeyPoint> puntosDetectados1, puntosDetectados2;
	Mat descriptores1, descriptores2;
	vector<DMatch> matches;
	
	//Creamos el matcher con FLANN.
	FlannBasedMatcher matcher;
	
	//Obtenemos los Key Points con ORB:
	puntosDetectados1 = obtenerKeyPointsORB(im1, 1000, ORB::HARRIS_SCORE, 35);
	puntosDetectados2 = obtenerKeyPointsORB(im2, 1000, ORB::HARRIS_SCORE, 35);
	
	
	//Obtenemos los descriptores de los puntos obtenidos en cada imagen.
	descriptores1 = obtenerDescriptoresORB(im1, 1000, ORB::HARRIS_SCORE, 35);
	descriptores2 = obtenerDescriptoresORB(im2, 1000, ORB::HARRIS_SCORE, 35);
	
	//Convertimos las matrices de descriptores a CV_32F para el correcto funcionamiento del matcher creado:
	descriptores1.convertTo(descriptores1, CV_32F);
	descriptores2.convertTo(descriptores2, CV_32F);
	
	
	matcher.match(descriptores1, descriptores2, matches);	
	
	return matches;
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
	matches = obtenerMatchesFuerzaBrutaBRISK(origen, destino, 65);
	
	//Obtenemos los puntos en correspondencias entre ambas imagenes:
	for (int i = 0; i < matches.size(); i++){
		puntosEnCorrespondenciasOrigen.push_back(puntosDetectadosOrigen[matches[i].queryIdx].pt);
		puntosEnCorrespondenciasDestino.push_back(puntosDetectadosDestino[matches[i].trainIdx].pt);	
	}
	
	//Calculamos la homografia con los puntos en correspondencias:	
	Mat H = findHomography(puntosEnCorrespondenciasOrigen, puntosEnCorrespondenciasDestino, CV_RANSAC);
	
	//Pasamos las homografia a 32F:
	H.convertTo(H, CV_32F);
	
	return H;
}


/*
Funcion que obtiene un mosaico de proyeccion plana de dos imagenes
@origen y destino: imagenes con las que formar el mosaico.
*/
Mat mosaicoDeDos (Mat origen, Mat destino) {
	//Creamos la imagen donde proyectaremos ambas imagenes:
	Mat mosaico = Mat(550, 1000, origen.type());
		
	//Colocamos la primera imagen en la esquina superior izquierda por medio de la identidad:
	Mat id = Mat(3,3,CV_32F,0.0);
	
	for (int i = 0; i < 3; i++)
		id.at<float>(i,i) = 1.0;
		
	warpPerspective(destino, mosaico, id, Size(mosaico.cols, mosaico.rows), INTER_LINEAR, BORDER_CONSTANT);
	
	//Calculamos la homografia que lleva la segunda imagen a la que hemos colocado primero en el plano de proyeccion:
	Mat homografia = calcularHomografia(origen, destino);
	
	//Colocamos la segunda imagen por medio de esa homografia (compuesta con la identidad):
	warpPerspective(origen, mosaico, homografia, Size(mosaico.cols, mosaico.rows), INTER_LINEAR, BORDER_TRANSPARENT);
	
	return mosaico;	
}

/*
Funcion que obtiene un mosaico de proyeccion plana con varias imagenes comenzando por la central en la realidad
@imagenes: imagenes para construir el mosaico
*/

Mat mosaicoDeN (vector<Mat> imagenes) {
	//Creamos la imagen donde formaremos el mosaico:
	Mat mosaico = Mat(700, 1100, imagenes.at(0).type());
	
	//Seleccionamos la posicion de la imagen central de la secuencia:
	int posicion_central = imagenes.size()/2;
	
	//Colocamos la imagen central del vector en el centro del mosaico
	Mat colocacionCentral = Mat(3,3,CV_32F,0.0);
	
	for (int i = 0; i < 3; i++)
		colocacionCentral.at<float>(i,i) = 1.0;
		
	//Realizamos una traslacion simplemente:
	colocacionCentral.at<float>(0,2) = mosaico.cols/2 - imagenes.at(posicion_central).cols/2;
	colocacionCentral.at<float>(1,2) = mosaico.rows/2 - imagenes.at(posicion_central).rows/2;
	
	warpPerspective(imagenes.at(posicion_central), mosaico, colocacionCentral, Size(mosaico.cols, mosaico.rows), INTER_LINEAR, BORDER_CONSTANT);
	
	//Matrices donde se acumularan las homografias a cada uno de los lados de la imagen central:
	Mat Hizda, Hdcha;
	//Las inicializamos con la homografia que hemos calculado antes, la que seria la H0:
	colocacionCentral.copyTo(Hizda);
	colocacionCentral.copyTo(Hdcha);
	
	/*
	Vamos formando el mosaico empezando desde la imagen central y desplazandonos a ambos extremos.
	En cada iteracion calculamos la homografia que queda cada imagen al mosaico componiendo con las que ya se han calculado para las imagenes anteriores.	
	*/
	for (int i = 1; i <= posicion_central; i++) {
		if (posicion_central-i >= 0){
			Hizda = Hizda * calcularHomografia(imagenes.at(posicion_central-i), imagenes.at(posicion_central-i+1));
			warpPerspective(imagenes.at(posicion_central-i), mosaico, Hizda, Size(mosaico.cols, mosaico.rows), INTER_LINEAR, BORDER_TRANSPARENT);
			
		}
		if (posicion_central+i < imagenes.size()){
			Hdcha = Hdcha * calcularHomografia(imagenes.at(posicion_central+i), imagenes.at(posicion_central+i-1));
			warpPerspective(imagenes.at(posicion_central+i), mosaico, Hdcha, Size(mosaico.cols, mosaico.rows), INTER_LINEAR, BORDER_TRANSPARENT);
		}
	}
	
	
	return mosaico;

}

/*
Funcion que obtiene un mosaico de proyeccion plana con varias imagenes comenzando por la primera de la secuencia
@imagenes: imagenes para construir el mosaico
*/
Mat mosaicoDeNMalo (vector<Mat> imagenes) {

	Mat mosaico = Mat(700, 1100, imagenes.at(0).type());
	
	//Colocamos la primera imagen en la parte izquierda del mosaico pero no arriba del todo:
	Mat colocacionInicial = Mat(3,3,CV_32F,0.0);
	
	for (int i = 0; i < 3; i++)
		colocacionInicial.at<float>(i,i) = 1.0;
		
	//Realizamos una traslacion simplemente:
	colocacionInicial.at<float>(1,2) = 100;
	
	warpPerspective(imagenes.at(0), mosaico, colocacionInicial, Size(mosaico.cols, mosaico.rows), INTER_LINEAR, BORDER_CONSTANT);
	
	//Matriz donde se acumularan las homografias segun vamos colocando imagenes en el mosaico:
	Mat H;
	//Se inicializa con la homografia calculada anteriormente:
	colocacionInicial.copyTo(H);
	
	/*
	Vamos recorriendo la secuencia de imagenes hacia la derecha.
	Calculamos la homografia que lleva la imagen actual a la anterior y la acumulamos (componemos) con las previamente calculadas.	
	*/
	for (int i = 1; i < imagenes.size(); i++) {
		H = H * calcularHomografia(imagenes.at(i), imagenes.at(i-1));
		warpPerspective(imagenes.at(i), mosaico, H, Size(mosaico.cols, mosaico.rows), INTER_LINEAR, BORDER_TRANSPARENT);	
	}	
	
	return mosaico;


}

void parte1() {
	cout << "Inicio Parte 1: " << endl;

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
	/*ptos_tablero1.push_back(Punto(157, 47));
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
	ptos_tablero2.push_back(Punto(131, 90));*/

	Mat H = obtenerMatrizTransformacion(ptos_tablero1, ptos_tablero2);
	Mat tablero1_transformada;

	warpPerspective(tablero1, tablero1_transformada, H, Size(tablero2.cols, tablero2.rows));

	vector<Mat> imagenes;

	imagenes.push_back(tablero1);
	imagenes.push_back(tablero1_transformada);
	imagenes.push_back(tablero2);


	mostrarImagenes("Calcular homografia (Aparado 1):", imagenes);
	
	waitKey(0);
	destroyAllWindows();



}

void parte2(Mat yose1, Mat yose2) {
	cout << "Inicio Parte 2: " << endl;
	
	vector<KeyPoint> puntosDetectados; //donde almacenaremos los puntos detectados para cada imagen por cada criterio, reutilizable.
	Mat yose1KPBRISK, yose1KPORB, yose2KPBRISK, yose2KPORB; //las imagenes correspondientes pintando los puntos detectados.

	puntosDetectados = obtenerKeyPointsBRISK(yose1, 65);
	cout << "En Yosemite1 con BRISK hemos obtenido: " << puntosDetectados.size() << " puntos" << endl;
	drawKeypoints(yose1, puntosDetectados, yose1KPBRISK);
	imshow("Yose 1 KP BRISK", yose1KPBRISK);
	puntosDetectados = obtenerKeyPointsBRISK(yose2, 65);
	cout << "En Yosemite2 con BRISK hemos obtenido: " << puntosDetectados.size() << " puntos" << endl;
	drawKeypoints(yose2, puntosDetectados, yose2KPBRISK);
	imshow("Yose 2 KP BRISK", yose2KPBRISK);
	puntosDetectados = obtenerKeyPointsORB(yose1, 1000, ORB::HARRIS_SCORE, 35);
	cout << "En Yosemite1 con ORB hemos obtenido: " << puntosDetectados.size() << " puntos" << endl;
	drawKeypoints(yose1, puntosDetectados, yose1KPORB);
	imshow("Yose 1 KP ORB", yose1KPORB);	
	puntosDetectados = obtenerKeyPointsORB(yose2, 1000, ORB::HARRIS_SCORE, 35);
	cout << "En Yosemite2 con ORB hemos obtenido: " << puntosDetectados.size() << " puntos" << endl;
	drawKeypoints(yose2, puntosDetectados, yose2KPORB);
	imshow("Yose 2 KP ORB", yose2KPORB);

	
	waitKey(0);
	destroyAllWindows();
}

void parte3(Mat yose1, Mat yose2) {
	cout << "Inicio Parte 3: " << endl;
	
	vector<KeyPoint> puntosDetectadosYose1, puntosDetectadosYose2;
	vector<DMatch> matchesFB, matchesFLANN;
	Mat imagenMatchesFBBRISK, imagenMatchesFBORB, imagenMatchesFLANNBRISK, imagenMatchesFLANNORB;
	
	puntosDetectadosYose1 = obtenerKeyPointsBRISK(yose1, 65);
	puntosDetectadosYose2 = obtenerKeyPointsBRISK(yose2, 65);

	matchesFB = obtenerMatchesFuerzaBrutaBRISK(yose1, yose2, 65);
	
	cout << "Se han obtenido " << matchesFB.size() << " matches con Fuerza Bruta + Cross check + BRISK" << endl;
	
	drawMatches( yose1, puntosDetectadosYose1, yose2, puntosDetectadosYose2, matchesFB, imagenMatchesFBBRISK, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
	
	imshow("Matches Fuerza Bruta con BRISK", imagenMatchesFBBRISK);
	
	matchesFLANN = obtenerMatchesFlannBRISK(yose1, yose2, 65);
	
	cout << "Se han obtenido " << matchesFLANN.size() << " matches con FLANN + BRISK" << endl;
	
	drawMatches( yose1, puntosDetectadosYose1, yose2, puntosDetectadosYose2, matchesFLANN, imagenMatchesFLANNBRISK, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
	
	
	imshow("Matches Flann con BRISK", imagenMatchesFLANNBRISK);
	
	puntosDetectadosYose1 = obtenerKeyPointsORB(yose1, 1000, ORB::HARRIS_SCORE, 35);
	puntosDetectadosYose2 = obtenerKeyPointsORB(yose2, 1000, ORB::HARRIS_SCORE, 35);

	matchesFB = obtenerMatchesFuerzaBrutaORB(yose1, yose2);
	
	cout << "Se han obtenido " << matchesFB.size() << " matches con Fuerza Bruta + Cross check + ORB" << endl;
	
	drawMatches( yose1, puntosDetectadosYose1, yose2, puntosDetectadosYose2, matchesFB, imagenMatchesFBORB, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
	
	imshow("Matches Fuerza Bruta con ORB", imagenMatchesFBORB);
	
	matchesFLANN = obtenerMatchesFlannORB(yose1, yose2);
	
	cout << "Se han obtenido " << matchesFLANN.size() << " matches con FLANN + ORB" << endl;
	
	drawMatches( yose1, puntosDetectadosYose1, yose2, puntosDetectadosYose2, matchesFLANN, imagenMatchesFLANNORB, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
	
	
	imshow("Matches Flann con ORB", imagenMatchesFLANNORB);
	
	waitKey(0);
	destroyAllWindows();
}

void parte4(Mat yose1, Mat yose2) {
	cout << "Inicio Parte 4: " << endl;
	
	Mat mosaico = mosaicoDeDos(yose2, yose1);

	imshow("Mosaico de 2", mosaico);
	
	waitKey(0);
	destroyAllWindows();

}

void parte5() {
	cout << "Inicio Parte 5: " << endl;

	vector<Mat> imagenes;
	
	//Secuencia buena:
	imagenes.push_back(imread("imagenes/mosaico002.jpg"));
	imagenes.push_back(imread("imagenes/mosaico003.jpg"));
	imagenes.push_back(imread("imagenes/mosaico004.jpg"));
	imagenes.push_back(imread("imagenes/mosaico006.jpg"));
	imagenes.push_back(imread("imagenes/mosaico005.jpg"));
	imagenes.push_back(imread("imagenes/mosaico007.jpg"));
	imagenes.push_back(imread("imagenes/mosaico008.jpg"));
	imagenes.push_back(imread("imagenes/mosaico009.jpg"));
	imagenes.push_back(imread("imagenes/mosaico010.jpg"));
	imagenes.push_back(imread("imagenes/mosaico011.jpg"));
	
	//Secuencia mala pero en el orden original:
	/*imagenes.push_back(imread("imagenes/mosaico002.jpg"));
	imagenes.push_back(imread("imagenes/mosaico003.jpg"));
	imagenes.push_back(imread("imagenes/mosaico004.jpg"));
	imagenes.push_back(imread("imagenes/mosaico005.jpg"));
	imagenes.push_back(imread("imagenes/mosaico006.jpg"));
	
	imagenes.push_back(imread("imagenes/mosaico007.jpg"));
	imagenes.push_back(imread("imagenes/mosaico008.jpg"));
	imagenes.push_back(imread("imagenes/mosaico009.jpg"));
	imagenes.push_back(imread("imagenes/mosaico010.jpg"));
	imagenes.push_back(imread("imagenes/mosaico011.jpg"));*/
	
	Mat mosaico = mosaicoDeN(imagenes);
	
	//Mat mosaico_malo = mosaicoDeNMalo(imagenes);
	
	imshow("Mosaico de varias imagenes", mosaico);
	
	waitKey(0);
	destroyAllWindows();

}



int main(int argc, char* argv[]) {

	cout << "OpenCV detectada " << endl;

/*
=====================================
PARTE 1: ESTIMACION DE LA HOMOGRAFIA
=====================================
*/

	parte1();

/*
==========================
PARTE 2: EXTRAER KEYPOINTS
==========================
*/

	//Cargamos las imagenes para los apartados 2 ,3 y 4:
	Mat yose1 = imread("imagenes/Yosemite1.jpg");
	Mat yose2 = imread("imagenes/Yosemite2.jpg");


	parte2(yose1, yose2);
	

/*
===============================
PARTE 3: DESCRIPTORES Y MATCHES
===============================
*/

	parte3(yose1, yose2);

/*
=================================
PARTE 4: MOSAICO CON DOS IMAGENES
=================================
*/

	parte4(yose1, yose2);
/*
====================================
PARTE 5: MOSAICO CON VARIAS IMAGENES
====================================
*/
	
	parte5();
	
	waitKey(0);
	destroyAllWindows();

	return 0;
}
