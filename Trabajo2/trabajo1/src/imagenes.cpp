#include<opencv2/opencv.hpp>
#include<iostream>
#include "imagenes.hpp"

using namespace std;
using namespace cv;

/*
Funcion que lee una imagen desde un archivo y devuelve el objeto Mat donde se almacena.
@ nombreArchivo: nombre del archivo desde el que se lee la imagen.
@ flagColor: opci?n de lectura de la imagen. Tenemos por defecto el valor 1 al igual que la funcion imread de OpenCV.
*/
Mat leeImagen(string nombreArchivo, int flagColor) {
	Mat imcargada = imread(nombreArchivo, flagColor); //leemos la imagen con la funcion imread con el flag especificado

													  //comprobamos si la lectura ha sido correcta, aunque devolvamos la matriz de todos modos
	if (imcargada.empty())
		cout << "Lectura incorrecta, se devolvera una matriz vacia." << endl;

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

/*
Funcion que modifica el color de los p?xeles indicados.
@ im = la imagen a la que le vamos a modificar los p?xeles.
@ coordenadas = lista de coordenadas de los p?xeles indicados.
@ color1, color2 y color3 = son tres colores, estamos pensando en trabajar con imagenes de 1 o 3 canales con lo que en cada caso se usara los colores necesarios.
*/
void modificarPixeles(Mat &im, vector<Point> &coordenadas, int color1, int color2, int color3) {
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