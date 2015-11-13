#include<opencv2/opencv.hpp>
#include<iostream>
#include "convolucion.hpp"

using namespace std;
using namespace cv;
/*
Funcion auxiliar para obtner valores de la funcion f a muestrear.
@x: el parametro a sustituir para la muestra.
*/

float f(float x, float sigma) {
	return exp(-0.5 * x * x / sigma / sigma);
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

	inicio_copia = pixels_extra / 2;
	fin_copia = colsParaCopia + inicio_copia;

	//Copiamos senal centrado en vectorAuxiliar

	for (int i = inicio_copia; i < fin_copia; i++)
		vectorOrlado.at<float>(0, i) = copia_senal.at<float>(0, i - inicio_copia);

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
Mat calcularConvolucionVectores1C(Mat &senal, Mat &mascara, int cond_contorno) {
	//preparamos el vector para la convolucion orlandolo.
	Mat copiaOrlada = obtenerVectorOrlado1C(senal, mascara, cond_contorno);
	Mat segmentoCopiaOrlada;
	Mat convolucion = Mat(1, senal.cols, senal.type());

	int inicio_copia, fin_copia, long_lado_orla;
	//calculamos el rango de pixeles a los que realmente tenemos que aplicar la convolucion, excluyendo los vectores de orla.
	inicio_copia = (mascara.cols - 1) / 2;
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
		Mat fila = im.row(i).clone();
		calcularConvolucionVectores1C(fila, mascara, cond_bordes).copyTo(convolucion.row(i));
	}

	//Convolucion por columnas
	convolucion = convolucion.t(); //trasponemos para poder operar como si fuese por filas

	for (int i = 0; i < convolucion.rows; i++) {
		Mat fila = convolucion.row(i);
		calcularConvolucionVectores1C(fila, mascara, cond_bordes).copyTo(fila);
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
