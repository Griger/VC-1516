#include<opencv2/opencv.hpp>
#include<vector>
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

/*
Funcion que lee una imagen desde un archivo y devuelve el objeto Mat donde se almacena.
@ nombreArchivo: nombre del archivo desde el que se lee la imagen.
@ flagColor: opci?n de lectura de la imagen. Tenemos por defecto el valor 1 al igual que la funcion imread de OpenCV.
*/
Mat leeImagen(string nombreArchivo, int flagColor = 1) {
	Mat imcargada = imread(nombreArchivo, flagColor); //leemos la imagen con la funcion imread con el flag especificado

													  //comprobamos si la lectura ha sido correcta, aunque devolvamos la matriz de todos modos
	if (imcargada.empty())
		cout << "Lectura incorrecta, se devolver? una matriz vacia." << endl;

	return imcargada;
}

/*
Funcion que lleva un @t en el rango [@a, @b] al rango [@c, @d] mediante una transformacion lineal.
*/
float cambioDeRango(float t, float a, float b, float c, float d) {
	return 1.0 * (t - a) / (b - a)*(d - c) + c;	
}

/*
Funcion que reajusta el rango de una matriz al rango [0,255] para que se muestren correctamente las frencuencias altas (tanto negativas como positivas)
@im: la imagen CV_32F a la que reajustaremos el rango.
*/

Mat reajustarRango1C(Mat im) {
	float min = 0;
	float max = 255;
	Mat im_ajustada;


	//Calculamos el rango en el que se mueven los valores de la imagen.
	for (int i = 0; i < im.rows; i++)
		for (int j = 0; j < im.cols; j++) {
			if (im.at<float>(i, j) < min) min = im.at<float>(i, j);
			if (im.at<float>(i, j) > max) max = im.at<float>(i, j);
		}

	im.copyTo(im_ajustada);

	for (int i = 0; i < im_ajustada.rows; i++)
		for (int j = 0; j < im_ajustada.cols; j++)
			im_ajustada.at<float>(i, j) = cambioDeRango(im_ajustada.at<float>(i, j), min, max, 0.0, 255.0);


	return im_ajustada;
}

/*
Funcion que reajusta el rango de una matriz al rango [0,255] para que se muestren correctamente las frencuencias altas (tanto negativas como positivas)
@im: la imagen CV_32F o CV_32FC3 a la que reajustaremos el rango.
*/
Mat reajustarRango(Mat im) {
	Mat canales_im[3];
	Mat im_ajustada;
	Mat canales_ajustada[3];


	if (im.channels() == 1)
		im_ajustada = reajustarRango1C(im);
	else if (im.channels() == 3) {
		split(im, canales_im);
		for (int i = 0; i < 3; i++)
			canales_ajustada[i] = reajustarRango(canales_im[i]);
		merge(canales_ajustada, 3, im_ajustada);
	}
	else
		cout << "El numero de canales no es correcto." << endl;

	return im_ajustada;
}

/*
Funcion que muestra una imagen por pantalla.
@ nombreVentana: el nombre de la ventana que crearemos y en la que mostraremos la imagen.
@ im: la imagen a mostrar
@ tipoVentana: el tipo de ventana que queremos crear, por defecto es 1, es decir una ventana que se ajusta al tamanio de la imagen.
*/
void mostrarImagen(string nombreVentana, Mat &im, int tipoVentana = 1) {
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

/*
Funcion que modifica el color de los p?xeles indicados.
@ im = la imagen a la que le vamos a modificar los p?xeles.
@ coordenadas = lista de coordenadas de los p?xeles indicados.
@ color1, color2 y color3 = son tres colores, estamos pensando en trabajar con imagenes de 1 o 3 canales con lo que en cada caso se usara los colores necesarios.
*/
void modificarPixeles(Mat &im, vector<Point> &coordenadas, int color1 = 0, int color2 = 0, int color3 = 0) {
	int y, x; //donde cargaremos las coordenadas del p?xel a modificar
	for (Point &p : coordenadas) {
		y = p.y; x = p.x;
		if (im.channels() == 1)
			im.at<uchar>(y, x) = color1; //si sólo tenemos un canal usamos el tipo uchar
		else if (im.channels() == 3) {
			im.at<Vec3b>(y, x)[0] = color1; //si hay tres canales usamos el tipo Vec3b pues seguimos con uchars pero ahora hemos de modificar tres componentes en cada píxel.
			im.at<Vec3b>(y, x)[1] = color2;
			im.at<Vec3b>(y, x)[2] = color3;
		}
		else
			cout << "Esta imagen no tiene un numero adecuado de canales." << endl;
	}
}

/*
Funcion auxiliar para obtner valores de la funcion f a muestrear.
@x: el parametro a sustituir para la muestra.
*/

float f(float x, float sigma) {
	return exp(-0.5 * x * x / sigma / sigma);
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
Funcion que calcula un vector mascara para realizar la convolucion en base a un sigma dado.
@sigma: parametro sigma (en unidades de pixeles) de la gaussiana a muestrear.
@f: la funcion de la que muestreamos para crear la mascara.
*/
Mat calcularVectorMascara(float sigma, float(*f)(float, float)) {
	/*
	Vamos a calcular el numero de pixeles que conformaran el vector mascara.
	En primer lugar recordemos que necesitamos tener una mascara de orden impar, de modos que haremoa a*2 + 1, donde a es la siguiente cantidad:
	Como sabemos para quedarnos con la parte significativa de la gaussiana tenemos que muestrear en el intervalor [-3sigma, 3sigma].
	Ahora bien, podemos obtener número decimales con lo que tendremos que redondear. Si hiciésemos la operación 3*ceil(sigma)*2+1 obtendríamos mascaras de como minimo longitud 7 algo que no es del todo adeacuado.
	Con lo cual vamos a optar por hacer un redondeo de la forma round(3*sigma)*2+1 y este sera el tamaño de nuestra mascara. Además hemos introudicido el 3 dentro del round porque de otro modo no seríamos tan preciso en el tamaño requerido,
	estaríamos en una situación similar a la anterior, mascaras de 1, luego de 7, sin ningun otro valor intermedio.
	Dejamos el 2 fuera del round para asegurar de obtener una longitud impar, como queremos.
	*/

	int longitud = round(3 * sigma) * 2 + 1;
	int centro = (longitud - 1) / 2; // <-- elemento del centro del vector

	/*
	Ahora vamos a calcular el tamanio del paso de muestre, como vamos a ir muestreando.
	Queremos que el mayor peso lo tenga el pixel central, con lo cual al este pixel le daremos el valor f(0).
	En consecuencia el paso será paso = 6sigma/(longitud-1).
	*/

	float paso = 6 * sigma / (longitud - 1);

	// Creamos la imagen que contendra los valores muestreados.

	Mat mascara = Mat(1, longitud, CV_32F);

	//Cargamos los valores en la mascara
	for (int i = 0; i <= centro; i++) {
		mascara.at<float>(0, i) = f(-paso * (centro - i), sigma);
		mascara.at<float>(0, longitud - i - 1) = f(paso * (centro - i), sigma);
	}

	//Y ahora dividimos por la suma de todos para que los elementos sumen 1.

	float suma = 0.0;

	for (int i = 0; i < mascara.cols; i++)
		suma += mascara.at<float>(0, i);

	mascara = mascara / suma;

	return mascara;
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


/*
Funcion que devuelve un vector preparado para hacer la convolucion sin problemas en los pixeles cercanos a los bordes, trabajando con imagenes con un solo canal.
@senal: vector de entrada al que aplicarle la convolucion.
@mascara: mascara con la que convolucionar (1C).
@cond_contorno: tipo de mecanismo para solucionar problemas con las dimensiones al convolucionar. Puede tomar dos valores:
				0 -> uniforme a ceros.
				1 -> reflejada.
*/
Mat obtenerVectorOrlado1C(Mat &senal, Mat &mascara, int cond_contorno) {
	//Añadiremos digamos a cada lado del vector (longitud_senal - 1)/2 pues son los pixeles como maximo que sobrarian al situar la mascara en una esquina.
	//Nosotros vamos a trabajar con vectores fila, pero no sabemos como sera senal con lo que vamos a trasponerla si es necesario.
	Mat copia_senal;

	//Vamos a trabajar siempre con vectores fila.
	if (senal.rows == 1)
		copia_senal = senal;
	else if (senal.cols == 1)
		copia_senal = senal.t();
	else
		cout << "Senal no es un vector fila o columna.\n";

	int colsParaCopia = copia_senal.cols;	
	
	int pixels_extra = mascara.cols - 1; //<-- numero de pixeles necesarios para orlar.

	int colsVectorOrlado = colsParaCopia + pixels_extra;

	Mat vectorOrlado = Mat(1, colsVectorOrlado, senal.type());

	int inicio_copia, fin_copia; // <-- posiciones donde comienza la copia del vector, centrada.

	inicio_copia = pixels_extra/2;
	fin_copia = colsParaCopia + inicio_copia;

	//Copiamos senal centrado en vectorAuxiliar

	for (int i = inicio_copia; i < fin_copia; i++)
		vectorOrlado.at<float>(0, i) = copia_senal.at<float>(0, i-inicio_copia);		
	
	// Ahora rellenamos los vectores de orlado segun la tecnica que hayamos elegido;
	// Hacemos el modo espejo solo que si la opcion elegida es cero entonces lo multiplicaremos por cero y en consecuencia sera el homogeneo a ceros.
	
	for (int i = 0; i < inicio_copia; i++) {
		vectorOrlado.at<float>(0, inicio_copia - i - 1) = cond_contorno * vectorOrlado.at<float>(0, inicio_copia + i);
		vectorOrlado.at<float>(0, fin_copia + i) = cond_contorno * vectorOrlado.at<float>(0, fin_copia - i - 1);
	}

	return vectorOrlado;

}


/*
Funcion que calcula la convolucion de dos vectores fila.
@senal: el vector al que le aplicamos la mascara de convolucion.
@mascara: la mascara de convolucion.
@cond_contorno: tipo de mecanismo para solucionar problemas con las dimensiones al convolucionar. Puede tomar dos valores:
				0 -> uniforme a ceros.
				1 -> reflejada.
*/
Mat calcularConvolucionVectores1C (Mat &senal, Mat &mascara, int cond_contorno) {
	//preparamos el vector para la convolucion orlandolo.
	Mat copiaOrlada = obtenerVectorOrlado1C(senal, mascara, cond_contorno);
	Mat segmentoCopiaOrlada;
	Mat convolucion = Mat(1, senal.cols, senal.type());

	int inicio_copia, fin_copia, long_lado_orla;
	//calculamos el rango de pixeles a los que realmente tenemos que aplicar la convolucion, excluyendo los vectores de orla.
	inicio_copia = (mascara.cols - 1)/2;
	fin_copia = inicio_copia + senal.cols;
	long_lado_orla = (mascara.cols - 1) / 2;

	for (int i = inicio_copia; i < fin_copia; i++) {
		//Vamos aplicando la convolucion a cada pixel seleccionando el segmento con el que convolucionamos.
		segmentoCopiaOrlada = copiaOrlada.colRange(i - long_lado_orla, i + long_lado_orla + 1);
		convolucion.at<float>(0, i - inicio_copia) = mascara.dot(segmentoCopiaOrlada);
	}

	return convolucion;
}

/*
Funcion que calcula la convolución de una imagen 1C con una mascara separada en un solo vector fila (por ser simetrica).
@im: la imagen CV_32F a convolucionar.
@sigma: el sigma de la mascara de convolucion.
@cond_bordes: la condicion con la que se orlan los vectores filas/columnas para prepararlos para la convolucion.
*/
Mat convolucion2D1C(Mat &im, float sigma, int cond_bordes) {
	Mat mascara = calcularVectorMascara(sigma, f); //calculamos la mascara a aplicar
	Mat convolucion = Mat(im.rows, im.cols, im.type()); //matriz donde introducimos el resultado de la convolucion

	//Convolucion por filas
	for (int i = 0; i < im.rows; i++) {
		calcularConvolucionVectores1C(im.row(i), mascara, cond_bordes).copyTo(convolucion.row(i));
	}

	//Convolucion por columnas
	convolucion = convolucion.t(); //trasponemos para poder operar como si fuese por filas

	for (int i = 0; i < convolucion.rows; i++) {
		calcularConvolucionVectores1C(convolucion.row(i), mascara, cond_bordes).copyTo(convolucion.row(i));
	}

	convolucion = convolucion.t(); //deshacemos la trasposicion para obtener el resultado final.

	return convolucion;
}


/*
Funcion que calcula la convolución de una imagen 1C o 3C con una mascara separada en un solo vector fila (por ser simetrica).
@im: la imagen CV_32F o CV_32FC3 a convolucionar.
@sigma: el sigma de la mascara de convolucion.
@cond_bordes: la condicion con la que se orlan los vectores filas/columnas para prepararlos para la convolucion.
*/
Mat convolucion2D(Mat &im, float sigma, int cond_bordes) {
	Mat convolucion;
	Mat canales[3];
	Mat canalesConvol[3];

	if (im.channels() == 1)
		return convolucion2D1C(im, sigma, cond_bordes);
	else if (im.channels() == 3) {
		split(im, canales);
		for (int i = 0; i < 3; i++)
			canalesConvol[i] = convolucion2D1C(canales[i], sigma, cond_bordes);
		merge(canalesConvol, 3, convolucion);
	}
	else
		cout << "Numero de canales no valido." << endl;

	return convolucion;
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
Funcion que obtiene la matriz de coeficientes para el sistema del calculo de la homografia dados los puntos muestreados en las imágenes estudiadas.
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
Obtenemos la matriz de trasnformacion que lleva una imagen a la otra, de forma aproximada:
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



int main(int argc, char* argv[]) {

	cout << "OpenCV detectada " << endl;	
	
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

	system("PAUSE");
	return 0;
}
