#include<opencv2/opencv.hpp>
#include<vector>
using namespace std;
using namespace cv;


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

	mostrarImagen(nombreVentana, collage);

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

/*TODO: Copiar aquí las funciones para muestrear*/


/*TODO: Los códigos son evidentes*/

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

Mat calcularConvolucionVectores1C (Mat &senal, Mat &mascara, int cond_contorno) {
	Mat copiaOrlada = obtenerVectorOrlado1C(senal, mascara, cond_contorno);
	Mat segmentoCopiaOrlada;
	Mat convolucion = Mat(1, senal.cols, senal.type());

	int inicio_copia, fin_copia, long_lado_orla;
	inicio_copia = (mascara.cols - 1)/2;
	fin_copia = inicio_copia + senal.cols; //poner para trasponer aqui
	long_lado_orla = (mascara.cols - 1) / 2;

	/*cout << "La copia de senal orlada es: " << endl;
	mostrarMatriz(copiaOrlada);
	cout << "La copia orlada tiene longitud: " << copiaOrlada.cols << endl;*/

	for (int i = inicio_copia; i < fin_copia; i++) {
		/*cout << "Segmento " << i - inicio_copia << endl;*/
		segmentoCopiaOrlada = copiaOrlada.colRange(i - long_lado_orla, i + long_lado_orla + 1);
		/*mostrarMatriz(segmentoCopiaOrlada);*/
		convolucion.at<float>(0, i - inicio_copia) = mascara.dot(segmentoCopiaOrlada);
	}
	
	/*cout << "El resultado de la convolucion es: " << endl;
	mostrarMatriz(convolucion);*/

	return convolucion;
}

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

/*
Funcion que calcula una imagen hibrida a partir de dos dadas.
@im1: una de las imagenes de las que partimos para formar la hibrida.
@im2: la otra imagen.
@sigma1: el sigma para la mascara de alisamiento a aplicar sobre im1.
@sigma2: el sigma para la mascara de alisamiento a aplicar sobre im2.
*/
Mat calcularImHibrida1C(Mat &im1, Mat &im2, float sigma1, float sigma2) {
	Mat frecuenciasBajas = convolucion2D(im1, sigma1, 0);
	Mat frecuenciasAltas = im2 - convolucion2D(im2, sigma2, 0);

	return frecuenciasAltas + frecuenciasBajas;
}

Mat calcularImHibrida(Mat &im1, Mat &im2, float sigma1, float sigma2) {
	Mat hibrida;
	Mat canalesIm1[3];
	Mat canalesIm2[3];
	Mat canalesHibrida[3];

	if (im1.channels() == 1 && im2.channels() == 1)
		hibrida = calcularImHibrida1C(im1, im2, sigma1, sigma2);
	else if (im1.channels() == 3 && im2.channels() == 3) {
		split(im1, canalesIm1);
		split(im2, canalesIm2);
		for (int i = 0; i < 3; i++)
			canalesHibrida[i] = calcularImHibrida1C(canalesIm1[i], canalesIm2[i], sigma1, sigma2);
		merge(canalesHibrida, 3, hibrida);
	}
	else
		cout << "Numero de canales no valido." << endl;

	return hibrida;
}

Mat submuestrear1C(Mat &im) {
	int colOriginal = im.cols;
	int filOriginal = im.rows;

	Mat submuestreado = Mat(filOriginal / 2, colOriginal / 2, im.type());

	for (int i = 0; i < submuestreado.rows; i++)
		for (int j = 0; j < submuestreado.cols; j++)
			submuestreado.at<float>(i, j) = im.at<float>(i*2+1, j*2+1);

	return submuestreado;
}

void calcularPirGaussiana1C(Mat &im, vector<Mat> &piramide, int numNiveles) {

	piramide.push_back(im);


	for (int i = 1; i < numNiveles; i++)
		piramide.push_back(submuestrear1C(convolucion2D1C(piramide.at(i-1), 0.5, 0)));
}

void calcularPirGaussiana(Mat &im, vector<Mat> &piramide, int numNiveles) {
	Mat canalesIm[3];
	Mat canalesNivel[3];
	vector<Mat> canalesPiramide[3];

	if (im.channels() == 1) 
		calcularPirGaussiana1C(im, piramide, numNiveles);	
	else if (im.channels() == 3) {
		piramide.resize(numNiveles);
		split(im, canalesIm);

		for (int i = 0; i < 3; i++)
			calcularPirGaussiana1C(canalesIm[i], canalesPiramide[i], numNiveles);
		

		for (int i = 0; i < numNiveles; i++) {
			for (int j = 0; j < 3; j++)
				canalesNivel[j] = canalesPiramide[j].at(i);

			merge(canalesNivel, 3, piramide.at(i));
		}

	}
	else
		cout << "Numero de canales no valido." << endl;
}

int main(int argc, char* argv[]) {

	cout << "OpenCV detectada " << endl;	
	
	int numNiveles = 6;

	Mat im1 = imread("imagenes/data/cat.bmp");
	Mat im2 = imread("imagenes/data/dog.bmp");
		
	im1.convertTo(im1, CV_32F);
	im2.convertTo(im2, CV_32FC3);
	
	Mat imhibrida = calcularImHibrida(im2, im1, 3.5, 10);

	vector<Mat> piramide;

	calcularPirGaussiana(imhibrida, piramide, numNiveles);

	for (int i = 0; i < numNiveles; i++) 
		piramide.at(i).convertTo(piramide.at(i), CV_8UC3);	

	mostrarImagenes("Piramide", piramide);
	
	waitKey();
	destroyAllWindows();

	system("PAUSE");
	return 0;
}
