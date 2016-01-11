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
			cout << m.at<double>(i,j) << " ";
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

	float longitud_x = (x_max - x_min);
	float longitud_y = (y_max - y_min);

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

	Mat imagen_ptos = Mat::zeros(500, 500, CV_32FC3);
	
	dibujarPtos(proyecciones_camara_original, imagen_ptos, Scalar(255, 0, 0));
	dibujarPtos(proyecciones_camara_estimada, imagen_ptos, Scalar(0, 255, 255));

	imshow("Ptos proyectados (azul) y ptos estimados (amarillo)", imagen_ptos);
	
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

	Size tamano_tablero = Size(13, 12);

	for (int i = 1; i <= 25; i++)
		imagenes_tablero.push_back( imread("imagenes/chessboard/Image"+to_string(i)+".tif", CV_8U));
		
	//Obtenemos las posiciones de las esquinas del tablero en las imagen donde podamos localizarlas.
	for (int i = 0; i < 25; i++) {
		if ( findChessboardCorners(imagenes_tablero.at(i), tamano_tablero, esquinas_img_actual) ) {
			imagenes_calibracion.push_back(imagenes_tablero.at(i));
			esquinas_imgs_calibracion.push_back(esquinas_img_actual);
		}

		esquinas_img_actual.clear();
	}

	cout << "Hemos podido localizar todas las esquinas en " << imagenes_calibracion.size() << " imagenes." << endl;
	
	//Refinamos las coordenadas obtenidas anteriormente.
	for (int i = 0; i < imagenes_calibracion.size(); i++) {
		//cout << "Hemos refinado las esquinas encontradas en la imagen " << i+1 << "/4" << endl;
		cornerSubPix( imagenes_calibracion.at(i), esquinas_imgs_calibracion.at(i), tamano_tablero, Size(-1,-1), TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ));
	}
	
	//Pintamos las esquinas encontradas:
	for (int i = 0; i < imagenes_calibracion.size(); i++) {
		cvtColor(imagenes_calibracion.at(i), imagenes_calibracion.at(i), CV_GRAY2BGR);
		drawChessboardCorners( imagenes_calibracion.at(i), tamano_tablero, Mat(esquinas_imgs_calibracion.at(i)), true);
	}
	
	imshow("Tablero 0", imagenes_calibracion.at(0));
	imshow("Tablero 1", imagenes_calibracion.at(1));
	imshow("Tablero 2", imagenes_calibracion.at(2));
	imshow("Tablero 3", imagenes_calibracion.at(3));
	
	vector<Point3f> esquinas_teoricas;

	//Obtenemos los ptos teoricos donde ha de estar el patron que estamos buscando
  for( int i = 0; i < tamano_tablero.height; i++)
    for( int j = 0; j < tamano_tablero.width; j++)
        esquinas_teoricas.push_back(Point3f(float(j), float(i), 0));

	vector<vector<Point3f> > puntos_objeto(1);
	puntos_objeto.resize(imagenes_calibracion.size(), esquinas_teoricas);




	waitKey(0);
	destroyAllWindows();


}

/*
Funcion que obtiene los KeyPoints de una imagen con el detector BRISK.
@im: imagen a la que le calculamos los KeyPoints
@umbral: parametro de umbral (thresh) para el detector BRISK a usar.
*/

vector<KeyPoint> obtenerKeyPoints (Mat im, int umbral = 30) {
	//Creamos el detector
	Ptr<BRISK> ptrDetectorBRISK = BRISK::create(umbral);
	vector<KeyPoint> puntosDetectados;
	
	//Obtenemos los KP:
	ptrDetectorBRISK->detect(im, puntosDetectados);

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
Funcion que calcula los puntos en correspondencias entre dos imagen por el criterio de Fuerza Bruta + comprobacion cruzada + BRISK
@im1 e im2: las imagenes entre las cuales vamos a buscar puntos en correspondencias.
@umbral: el umbral para el detector BRISK.
*/
vector<DMatch> obtenerMatches (Mat im1, Mat im2, int umbral){
	Mat descriptores1, descriptores2;
	vector<DMatch> matches;
	
	//Creamos el matcher con Fuerza Bruta activandole el flag para el cross check.
	BFMatcher matcher = BFMatcher(NORM_L2, true);
	
	//Obtenemos los descriptores de los puntos obtenidos en cada imagen.
	descriptores1 = obtenerDescriptoresBRISK(im1, umbral);
	descriptores2 = obtenerDescriptoresBRISK(im2, umbral);
	
	//clock_t t_inicio= clock();
	//Calculamos los matches entre ambas imagenes:
	matcher.match(descriptores1, descriptores2, matches);		
	//printf("FB ha tardado: %.2fs\n",(double)(clock() - t_inicio)/CLOCKS_PER_SEC);
	
	return matches;
}

Mat estimarF (Mat im1, Mat im2, int umbral, vector<Point2f> &corresp_1, vector<Point2f> &corresp_2) {
	vector<KeyPoint> KPim1 = obtenerKeyPoints(im1, umbral);
	vector<KeyPoint> KPim2 = obtenerKeyPoints(im2, umbral);
	vector<DMatch> matches = obtenerMatches(im1, im2, umbral);
	
	//Construimos los vectores de ptos en correspondencias para el calculo de F
	vector<Point2f> ptos_corresp_1, ptos_corresp_2;
	
	for (int i = 0; i < matches.size(); i++) {
		ptos_corresp_1.push_back(KPim1[matches[i].queryIdx].pt);
		ptos_corresp_2.push_back(KPim2[matches[i].trainIdx].pt);
	}
	
	//Calculamos la matriz fundamental:
	vector <unsigned char> buenos_malos; //vector donde se marca que parejas han sido rechazadas por RANSAC
	Mat F = findFundamentalMat(ptos_corresp_1, ptos_corresp_2, CV_FM_8POINT + CV_FM_RANSAC, 1, 0.99, buenos_malos);
	
	vector<Vec3f> lineas_para_1, lineas_para_2;
	computeCorrespondEpilines(ptos_corresp_1, 1, F, lineas_para_1);
	computeCorrespondEpilines(ptos_corresp_2, 2, F, lineas_para_2);
	
	double error1 = 0;
	double error2 = 0;
	int dem = 0;
	Vec3f l;
	Point2f p;
	for (int i = 0; i < lineas_para_1.size(); i++) {
		if (buenos_malos.at(i) == 1) {
			l = lineas_para_1.at(i);
			p = ptos_corresp_2.at(i);
			corresp_2.push_back(p);
			//cout << abs(l[0]*p.x + l[1]*p.y + l[2]) / sqrt(l[0]*l[0]+l[1]*l[1]) << " ";
			error1 += abs(l[0]*p.x + l[1]*p.y + l[2]) / sqrt(l[0]*l[0]+l[1]*l[1]);
		
			l = lineas_para_2.at(i);
			p = ptos_corresp_1.at(i);
			corresp_1.push_back(p);
			//cout << abs(l[0]*p.x + l[1]*p.y + l[2]) / sqrt(l[0]*l[0]+l[1]*l[1]) << " ";
			error2 += abs(l[0]*p.x + l[1]*p.y + l[2]) / sqrt(l[0]*l[0]+l[1]*l[1]);
			
			dem++;
		}
	}
	
	error1 = error1 / dem;
	error2 = error2 / dem;
	
	cout << "El error promedio cometido para las lineas de 1 es: " << error1 << endl;
	cout << "El error promedio cometido para las lineas de 2 es: " << error2 << endl;	
	
	
	
	return F;
}
	

//Funcion donde se estructuran los pasos necesarios para el tercer punto de la practica
void parte3() {
	//Cargamos las imagenes
	Mat vmort1 = imread("imagenes/vmort/Vmort1.pgm");
	Mat vmort2 = imread("imagenes/vmort/Vmort2.pgm");
	
	int umbral = 97;
	vector<KeyPoint> KPvmort1 = obtenerKeyPoints(vmort1, umbral);
	vector<KeyPoint> KPvmort2 = obtenerKeyPoints(vmort2, umbral);
	vector<DMatch> matches = obtenerMatches(vmort1, vmort2, umbral);
	
	cout << "Hemos obtenido: " << KPvmort1.size() << " y " << KPvmort2.size() << "Key Points" << endl;
	cout << "Y de aqui hemos obtenido: " << matches.size() << "parejas en correspondencias. " << endl;
	
	//Construimos los vectores de ptos en correspondencias para el calculo de F.
	vector<Point2f> ptosCorrespondenciasvmort1, ptosCorrespondenciasvmort2;
		
	for (int i = 0; i < matches.size(); i++){
		ptosCorrespondenciasvmort1.push_back(KPvmort1[matches[i].queryIdx].pt);
		ptosCorrespondenciasvmort2.push_back(KPvmort2[matches[i].trainIdx].pt);	
	}
	
	//Calculamos la matriz fundamental:
	vector<unsigned char> buenos_malos;
	Mat F = findFundamentalMat(ptosCorrespondenciasvmort1, ptosCorrespondenciasvmort2, CV_FM_8POINT+CV_FM_RANSAC,1,0.99, buenos_malos);
	cout << "Se ha estimado la matriz fundamental y es:" << endl;
	mostrarMatriz(F);
	
	int numero_descartes = 0;
	
	for (int i = 0; i < buenos_malos.size(); i++)
		if (buenos_malos.at(i) == 0)
			numero_descartes++;
		
	cout << "RANSAC ha descartado: " << numero_descartes << " parejas en correspondencias." << endl;
	
	vector<Vec3f> lineas_para_vmort1, lineas_para_vmort2;
	computeCorrespondEpilines(ptosCorrespondenciasvmort1, 1, F, lineas_para_vmort1);
	computeCorrespondEpilines(ptosCorrespondenciasvmort2, 2, F, lineas_para_vmort2);
	
	cout << "Se han obtenido: " << lineas_para_vmort1.size() << "lineas epipolares" << endl;
	Vec3f l;
	double c = vmort2.cols;
	
	//Dibujamos las lineas epipolares evaluandolas en x = 0 y x = num_columnas_imagen
	for (int i = 0; i < lineas_para_vmort1.size(); i++) {
		if (buenos_malos.at(i) == 1) {
			l = lineas_para_vmort1.at(i);
			line(vmort2, Point(0, -l[2]/l[1]), Point(c, (-l[2]-l[0]*c)/l[1]), CV_RGB(rand() % 256,rand() % 256 ,rand() % 256));
		}	
	}
	
	c = vmort1.cols;
	
	for (int i = 0; i < lineas_para_vmort2.size(); i++) {
		if (buenos_malos.at(i) == 1) {
			l = lineas_para_vmort2.at(i);
			line(vmort1, Point(0, -l[2]/l[1]), Point(c, (-l[2]-l[0]*c)/l[1]), CV_RGB(rand() % 256,rand() % 256 ,rand() % 256));
		}	
	}
	
	imshow("Epipolares de los ptos de Vmort2 sobre Vmort1", vmort1);	
	imshow("Epipolares de los ptos de Vmort1 sobre Vmort2", vmort2);
	
	Point2f p;
	//Calculamos el error como las distancia promedio de las lineas epipolares a sus puntos de soporte
	double error1 = 0;
	double error2 = 0;
	int dem = 0;
	for (int i = 0; i < lineas_para_vmort1.size(); i++) {
		if (buenos_malos.at(i) == 1) {
			l = lineas_para_vmort1.at(i);
			p = ptosCorrespondenciasvmort2.at(i);
			//cout << abs(l[0]*p.x + l[1]*p.y + l[2]) / sqrt(l[0]*l[0]+l[1]*l[1]) << " ";
			error1 += abs(l[0]*p.x + l[1]*p.y + l[2]) / sqrt(l[0]*l[0]+l[1]*l[1]);
		
			l = lineas_para_vmort2.at(i);
			p = ptosCorrespondenciasvmort1.at(i);
			//cout << abs(l[0]*p.x + l[1]*p.y + l[2]) / sqrt(l[0]*l[0]+l[1]*l[1]) << " ";
			error2 += abs(l[0]*p.x + l[1]*p.y + l[2]) / sqrt(l[0]*l[0]+l[1]*l[1]);
			
			dem++;
		}
	}
	
	error1 = error1 / dem;
	error2 = error2 / dem;
	
	cout << "El error promedio cometido para las lineas de vmort1 es: " << error1 << endl;
	cout << "El error promedio cometido para las lineas de vmort2 es: " << error2 << endl;	
		
	
	waitKey(0);
	destroyAllWindows();

}

//Funcion donde se estructuran los pasos necesarios para el cuarto punto de la practica
void parte4() {
	//Cargamos las imagenes
	Mat im0 = imread("imagenes/reconstruccion/rdimage.000.ppm");
	Mat im1 = imread("imagenes/reconstruccion/rdimage.001.ppm");
	Mat im4 = imread("imagenes/reconstruccion/rdimage.004.ppm");
	
	//Recuperamos los puntos en correspondencias que se han usado en la estimacion de F
	vector<Point2f> corresp_1, corresp_2;

	Mat F = estimarF(im0, im1, 50, corresp_1, corresp_2);
	
	cout << "Se ha calculado la matriz fundamental y es: " << endl;
	mostrarMatriz(F);
	//cout << "El tipo de la matriz F es: " << F.type() << endl;
	Mat K = Mat(3,3,CV_64F);
	K.at<double>(0,0) = 1839.6300000000001091;
	K.at<double>(0,1) = 0.0;
	K.at<double>(0,2) = 1024.2000000000000455;
	K.at<double>(1,0) = 0.0;
	K.at<double>(1,1) = 1848.0699999999999363;
	K.at<double>(1,2) = 686.5180000000000291;
	K.at<double>(2,0) = 0.0;
	K.at<double>(2,1) = 0.0;
	K.at<double>(2,2) = 1.0;

	cout << "La matriz de parametros intrinsecos es: " << endl;
	mostrarMatriz(K);

	//Estimamos la matriz esencial:
	Mat E1 = K.t() * F;
	Mat E = E1 * K;
	
	cout << "Hemos estimado E: " << endl;
	mostrarMatriz(E);
	
	Mat EEt = E*E.t();
	
	double traza = 0.0;
	for (int i = 0; i < 3; i++)
		traza += EEt.at<double>(i,i);

	cout << "Traza de EEtrapuesta: " << traza << endl;
	
	Mat E_norm = E / sqrt(traza/2);

	Mat EEt_norm = E_norm * E_norm.t();
	

	cout << "EEtrapuesta normalizada: " << endl;
	mostrarMatriz(EEt_norm);
	
	Mat T = Mat(1,3, CV_64F);
	Mat menos_T = Mat(1,3, CV_64F);
	int fila_donde_despejar;

	double elem = EEt_norm.at<double>(0,0);
	for (int i = 0; i < 3; i++)
		if (EEt_norm.at<double>(i,i) <= elem) {
			fila_donde_despejar = i;
			elem = EEt_norm.at<double>(i,i);
		}

	T.at<double>(0, fila_donde_despejar) = sqrt(1-elem);
	T.at<double>(0,(fila_donde_despejar+1)%3) = -EEt_norm.at<double>(fila_donde_despejar, (fila_donde_despejar+1)%3) / sqrt(1-elem);
	T.at<double>(0,(fila_donde_despejar+2)%3) = -EEt_norm.at<double>(fila_donde_despejar, (fila_donde_despejar+2)%3) / sqrt(1-elem);
	
	menos_T.at<double>(0,0) = -T.at<double>(0,0);
	menos_T.at<double>(0,1) = -T.at<double>(0,1);
	menos_T.at<double>(0,2) = -T.at<double>(0,2);
	
	Mat menos_E_norm = -E_norm;
	//mostrarMatriz(menos_T);
	Mat R_E_T = Mat(3,3,CV_64F);
	Mat R_E_menosT = Mat(3,3,CV_64F);
	Mat R_menosE_T = Mat(3,3,CV_64F);
	Mat R_menosE_menosT = Mat(3,3,CV_64F);
	
	(E_norm.row(0).cross(T)).copyTo(R_E_T.row(0));
	(E_norm.row(1).cross(T)).copyTo(R_E_T.row(1));
	(E_norm.row(2).cross(T)).copyTo(R_E_T.row(2));
	
	(E_norm.row(0).cross(menos_T)).copyTo(R_E_menosT.row(0));
	(E_norm.row(1).cross(menos_T)).copyTo(R_E_menosT.row(1));
	(E_norm.row(2).cross(menos_T)).copyTo(R_E_menosT.row(2));
	
	(menos_E_norm.row(0).cross(T)).copyTo(R_menosE_T.row(0));
	(menos_E_norm.row(1).cross(T)).copyTo(R_menosE_T.row(1));
	(menos_E_norm.row(2).cross(T)).copyTo(R_menosE_T.row(2));
	
	(menos_E_norm.row(0).cross(menos_T)).copyTo(R_menosE_menosT.row(0));
	(menos_E_norm.row(1).cross(menos_T)).copyTo(R_menosE_menosT.row(1));
	(menos_E_norm.row(2).cross(menos_T)).copyTo(R_menosE_menosT.row(2));
	
	cout << "La rotacion para E y T:" << endl;
	mostrarMatriz(R_E_T);
	
	cout << "La rotacion para E y -T:" << endl;
	mostrarMatriz(R_E_menosT);
	
	cout << "La rotacion para -E y T:" << endl;
	mostrarMatriz(R_menosE_T);
	
	cout << "La rotacion para -E y -T:" << endl;
	mostrarMatriz(R_menosE_menosT);
	
	//Obtenemos la distancia focal en pixels de la matriz de calibracion K:
	double f = K.at<double>(0,0);
	
	int num_corresp = corresp_1.size();
	double Zis[num_corresp];
	double dot1, dot2, Zi, Zd;
	Mat pi = Mat(1,3,CV_64F);
	Mat Pi = Mat(1,3,CV_64F);
	pi.at<double>(0,2) = 1.0;
	/*Point2f p = corresp_1.at(0);
	
	cout << "El p es (" << p.x << "," << p.y << ")" << endl;
	pi.at<double>(0,0) = corresp_1.at(0).x;
	pi.at<double>(0,1) = corresp_1.at(0).y;
	
	cout << "pi" << endl;
	mostrarMatriz(pi);*/
	//cout << "El T es: "; mostrarMatriz(T);
	//cout << "La f es: " << f << endl;
	Mat R = R_E_menosT;
	int contador = 0;
	for (int i = 0; i < corresp_1.size(); i++) {
		pi.at<double>(0,0) = corresp_1.at(i).x;
		pi.at<double>(0,1) = corresp_1.at(i).y;
		//cout << "xd: " << corresp_2.at(i).x << endl;
		//mostrarMatriz(f*R_E_T.row(0) - corresp_2.at(i).x*R_E_T.row(2));
		dot1 = (f*R.row(0) - corresp_2.at(i).x*R.row(2)).dot(menos_T);
		//cout << "El producto escalar vale: " << dot << endl;
		dot2 = (f*R.row(0) - corresp_2.at(i).x*R.row(2)).dot(pi);
		Zi = f*dot1/dot2;
		
		Pi = (Zi/f)*pi;
		
		Zd = R.row(2).dot(Pi-menos_T);
		
		if (Zi < 0 || Zd < 0)
			contador++;
		
	}
	
	cout << "Hay " << contador << " Zs negativos de" << corresp_1.size() << " posibles." << endl;
		
		

	waitKey(0);
	destroyAllWindows();




}


int main() {
	//cout << "*****************************\nPARTE 1: ESTIMACION DE CAMARA\n*****************************" << endl;
	//parte1();

	//cout << "********************\nPARTE 2: CALIBRACION\n********************" << endl;
	//parte2();
	
	//cout << "************************\nPARTE 3: ESTIMACION DE F\n************************" << endl;
	//parte3();
	
	cout << "*********************************\nPARTE 4: ESTIMACION DE MOVIMIENTO\n*********************************" << endl;
	parte4();
}
