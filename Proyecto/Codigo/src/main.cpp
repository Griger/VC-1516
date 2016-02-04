#include<opencv2/opencv.hpp>
#include<string>
#include<cmath>

using namespace std;
using namespace cv;

/*
Funcion que lleva un @t en el rango [@a, @b] al rango [@c, @d] mediante una transformacion lineal.
*/
float rankChange(float t, float a, float b, float c, float d) {
	return 1.0 * (t - a) / (b - a)*(d - c) + c;
}

/*
Funcion que reajusta el rango de una matriz al rango [0,255] para que se muestren correctamente las frencuencias altas (tanto negativas como positivas)
@im: la imagen CV_32F a la que reajustaremos el rango.
*/

Mat adjustRank1C(Mat im) {
	float min = 0;
	float max = 255;
	Mat adjusted_im;


	//Calculamos el rango en el que se mueven los valores de la imagen.
	for (int i = 0; i < im.rows; i++)
		for (int j = 0; j < im.cols; j++) {
			if (im.at<float>(i, j) < min) min = im.at<float>(i, j);
			if (im.at<float>(i, j) > max) max = im.at<float>(i, j);
		}

	im.copyTo(adjusted_im);

	for (int i = 0; i < adjusted_im.rows; i++)
		for (int j = 0; j < adjusted_im.cols; j++)
			adjusted_im.at<float>(i, j) = rankChange(adjusted_im.at<float>(i, j), min, max, 0.0, 255.0);


	return adjusted_im;
}


/*
Funcion que devuelve un vector preparado para hacer la convolucion sin problemas en los pixeles cercanos a los bordes, trabajando con imagenes con un solo canal.
@signal: vector de entrada al que aplicarle la convolucion.
@mask: mascara con la que convolucionar (1C).
*/
Mat getEdged1CVector (Mat &signal, Mat &mask) {
	//A�adiremos digamos a cada lado del vector (longitud_senal - 1)/2 pues son los pixeles como maximo que sobrarian al situar la mascara en una esquina.
	//Nosotros vamos a trabajar con vectores fila, pero no sabemos como sera senal con lo que vamos a trasponerla si es necesario.
	Mat signal_copy = signal;

	int copy_cols = signal_copy.cols;

	int extra_pixels = 4; //<-- numero de pixeles necesarios para orlar.

	int edged_vector_cols = copy_cols + extra_pixels;

	Mat edged_vector = Mat(1, edged_vector_cols, signal.type());

	int copy_start, copy_end; // <-- posiciones donde comienza la copia del vector, centrada.

	copy_start = extra_pixels/2;
	copy_end = copy_cols + copy_start - 1;

	//Copiamos senal centrado en vectorAuxiliar

	for (int i = copy_start; i <= copy_end; i++)
		edged_vector.at<float>(0, i) = signal_copy.at<float>(0, i-copy_start);

	// Ahora rellenamos los vectores de orlado segun la tecnica del articulo.
	for (int i = 1; i <= copy_start; i++) {
		edged_vector.at<float>(0, copy_start - i) = 2 * edged_vector.at<float>(0, copy_start) - edged_vector.at<float>(0, copy_start + i);
		edged_vector.at<float>(0, copy_end + i) = 2 * edged_vector.at<float>(0, copy_end) - edged_vector.at<float>(0, copy_end -i);
	}

	return edged_vector;
}


/*
Funcion que calcula la convolucion de dos vectores fila.
@signal: el vector al que le aplicamos la mascara de convolucion.
@mask: la mascara de convolucion.
*/
Mat computeConvolution1CVectors (Mat signal, Mat mask) {
	//preparamos el vector para la convolucion orlandolo.
	Mat edged_copy = getEdged1CVector(signal, mask);
	Mat edged_copy_segment;
	Mat convolution = Mat(1, signal.cols, signal.type());

	//cout << "Hemos obtenido ya el vector orlado y tiene " << edged_copy.rows << " filas y " << edged_copy.cols << " columnas"<< endl;

	int copy_start, copy_end, border_side_length;
	//calculamos el rango de pixeles a los que realmente tenemos que aplicar la convolucion, excluyendo los vectores de orla.
	copy_start = (mask.cols - 1)/2;
	copy_end = copy_start + signal.cols;
	border_side_length = (mask.cols - 1) / 2;

	for (int i = copy_start; i < copy_end; i++) {
		//Vamos aplicando la convolucion a cada pixel seleccionando el segmento con el que convolucionamos.
		edged_copy_segment = edged_copy.colRange(i - border_side_length, i + border_side_length + 1);
		convolution.at<float>(0, i - copy_start) = mask.dot(edged_copy_segment);
	}

	//cout << "Devolvemos la convolucion de los dos vectores" << endl;
	return convolution;
}
/*
Funcion que calcula la convolucion de una imagen 1C con una mascara separada en un solo vector fila (por ser simetrica).
@im: la imagen CV_32F a convolucionar.
*/
Mat convolution2D1C (Mat im) {
	Mat mask = Mat::zeros(1, 5, CV_32F);
	mask.at<float>(0,0) = 0.05;
	mask.at<float>(0,1) = 0.25;
	mask.at<float>(0,2) = 0.4;
	mask.at<float>(0,3) = 0.25;
	mask.at<float>(0,4) = 0.05;

	Mat convolution = Mat(im.rows, im.cols, im.type()); //matriz donde introducimos el resultado de la convolucion

	//cout << "Empezamos la convolucion por filas: " << endl;
	//Convolucion por filas
	for (int i = 0; i < convolution.rows; i++) {
		computeConvolution1CVectors(im.row(i), mask).copyTo(convolution.row(i));
	}

	//Convolucion por columnas
	convolution = convolution.t(); //trasponemos para poder operar como si fuese por filas

	//cout << "Empezamos la convolucion por columnas: " << endl;
	for (int i = 0; i < convolution.rows; i++) {
		computeConvolution1CVectors(convolution.row(i), mask).copyTo(convolution.row(i));
	}

	convolution = convolution.t(); //deshacemos la trasposicion para obtener el resultado final.

	return convolution;
}

/*
Funcion que submuestrea una imagen tomando solo las columnas y filas impares.
@im: la imagen CV_32F a submuestrear.
*/
Mat subSample1C(Mat &im) {
	Mat subsample = Mat(im.rows / 2, im.cols / 2, im.type());

	for (int i = 0; i < subsample.rows; i++)
		for (int j = 0; j < subsample.cols; j++)
			subsample.at<float>(i, j) = im.at<float>(i*2, j*2);

	return subsample;
}

Mat reduce(Mat im){
	//cout << "Vamos a hacer la convolucion " << endl;
	Mat convolution = convolution2D1C(im);

	//cout << "Vamos a hacer el submuestreado " << endl;
	return subSample1C(convolution);
}

vector<Mat> computeGaussianPyramid(Mat image){
	vector<Mat> gaussianPyramid;
	Mat actualLevelMatrix = image;
	Mat copy_actual_level;

	while (5 <= actualLevelMatrix.cols && 5 <= actualLevelMatrix.rows){
		//cout << "El nivel actual tiene: " << actualLevelMatrix.rows << " filas y " << actualLevelMatrix.cols << " columnas." << endl;
		gaussianPyramid.push_back(actualLevelMatrix);
		actualLevelMatrix.copyTo(copy_actual_level);
		//cout << "Vamos a hacer el reduce: " << endl;
		actualLevelMatrix = reduce(copy_actual_level);
	}

	return gaussianPyramid;
}

/*
Funcion que orla la matriz para poder hacer la operacion expand
@im: la matriz a orlar
*/
Mat getEdgedMat1C (Mat im) {
	Mat edged_mat = Mat::zeros(im.rows+2, im.cols+2, im.type());

	//Copiamos im dentro de la nueva matriz dejando el borde
	for (int i = 1; i < edged_mat.rows - 1; i++)
		for (int j = 1; j < edged_mat.cols - 1 ; j++)
			edged_mat.at<float>(i,j) = im.at<float>(i-1, j-1);

	/*cout << "La orlada es: " << endl;
	Mat orlada;
	edged_mat.convertTo(orlada, CV_8U);
	imshow("Orlada", orlada);*/

	for (int i = 0; i < edged_mat.rows; i++) {
		edged_mat.at<float>(i, 0) = 2*edged_mat.at<float>(i, 1) - edged_mat.at<float>(i, 2);
		edged_mat.at<float>(i, edged_mat.cols - 1) = 2*edged_mat.at<float>(i, edged_mat.cols -2) - edged_mat.at<float>(i, edged_mat.cols -3);
	}

	for (int j = 0; j < edged_mat.cols; j++) {
		edged_mat.at<float>(0, j) = 2*edged_mat.at<float>(1, j) - edged_mat.at<float>(2, j);
		edged_mat.at<float>(edged_mat.rows - 1, j) = 2*edged_mat.at<float>(edged_mat.rows - 2, j) - edged_mat.at<float>(edged_mat.rows - 3, j);
	}

	edged_mat.at<float>(0,0) = 2*edged_mat.at<float>(1, 1) - edged_mat.at<float>(2, 2);
	edged_mat.at<float>(0,edged_mat.cols - 1) = 2*edged_mat.at<float>(1, edged_mat.cols - 2) - edged_mat.at<float>(2, edged_mat.cols - 3);
	edged_mat.at<float>(edged_mat.rows - 1,0) = 2*edged_mat.at<float>(edged_mat.rows - 2, 1) - edged_mat.at<float>(edged_mat.rows - 3, 2);
	edged_mat.at<float>(edged_mat.rows - 1, edged_mat.cols - 1) = 2*edged_mat.at<float>(edged_mat.rows - 2, edged_mat.cols - 2) - edged_mat.at<float>(edged_mat.rows - 3, edged_mat.cols - 3);

	return edged_mat;
}

/*
Funcion que devuelve el valor w(m,n) siendo w la funcion de pesos con m,n en el conjunto {-2,-1,0,1,2}
*/
float w (int m, int n) {
	Mat w_hat = Mat::zeros(1, 5, CV_32F);
	w_hat.at<float>(0,0) = 0.05;
	w_hat.at<float>(0,1) = 0.25;
	w_hat.at<float>(0,2) = 0.4;
	w_hat.at<float>(0,3) = 0.25;
	w_hat.at<float>(0,4) = 0.05;

	return w_hat.at<float>(0, m+2)*w_hat.at<float>(0,n+2);
}

/*
Funcion que devuelve el elemento (@i, @j) de la matriz resultante de hacer expand(@prev_level)
*/
float getExpansionValue (int i, int j, Mat prev_level) {
	float sum = 0.0;

	for (int m = -2; m <= 2; m++)
		for (int n = -2; n <= 2; n++)
			if ( (i-m+1)%2 == 0 && (j-n+1)%2 == 0 )
				sum += w(m,n)*prev_level.at<float>((i-m+1)/2, (j-n+1)/2);

	return 4*sum;
}

/*
Funcion que realiza la operacion de expansion sobre una matriz
@im: la matriz a expandir
@result_rows: filas que tiene el siguiente nivel de la laplaciana
@result_cols: columnas que tiene el siguiente nivel de la laplaciana
*/
Mat expand (Mat im, int result_rows, int result_cols) {

	Mat edged_im = getEdgedMat1C(im);
	Mat expansion = Mat::zeros(result_rows, result_cols, CV_32F);

	for (int i = 0; i < result_rows; i++)
		for (int j = 0; j < result_cols; j++)
			expansion.at<float>(i,j) = getExpansionValue(i,j, edged_im);

	return expansion;
}

vector<Mat> computeLaplacianPyramid(Mat image){
	vector<Mat> solution, gaussianPyramid;
	Mat actualLevelMatrix;

	gaussianPyramid = computeGaussianPyramid(image);

	for (int i = 0; i < gaussianPyramid.size(); i++){
		if(i < gaussianPyramid.size() - 1)
			actualLevelMatrix = gaussianPyramid.at(i) - expand(gaussianPyramid.at(i+1), gaussianPyramid.at(i).rows, gaussianPyramid.at(i).cols);
		else
			actualLevelMatrix = gaussianPyramid.at(i);

		solution.push_back(actualLevelMatrix);
	}

	return solution;
}

vector<Mat> combineLaplacianPyramids(vector<Mat> laplacian_pyramidA, vector<Mat> laplacian_pyramidB, vector<Mat> mask_gaussian_pyramid){

	vector<Mat> combined_pyramids;
	Mat actual_level_matrix;

	for (int k = 0; k < laplacian_pyramidA.size(); k++){
		actual_level_matrix = Mat::zeros(laplacian_pyramidA.at(k).rows,laplacian_pyramidA.at(k).cols, CV_32F);

		for(int i = 0; i < laplacian_pyramidA.at(k).rows; i++)
			for(int j = 0; j < laplacian_pyramidA.at(k).cols; j++)
				actual_level_matrix.at<float>(i,j) += laplacian_pyramidA.at(k).at<float>(i,j) * mask_gaussian_pyramid.at(k).at<float>(i,j) + (1-mask_gaussian_pyramid.at(k).at<float>(i,j)) * laplacian_pyramidB.at(k).at<float>(i,j);

		combined_pyramids.push_back(actual_level_matrix);
	}

	return combined_pyramids;
}

Mat restoreImageFromLP (vector<Mat> laplacian_pyramid) {
	Mat reconstruction;
	vector<Mat> reconstructions;

	int levels_num = laplacian_pyramid.size();
	reconstructions.push_back(laplacian_pyramid.at(levels_num-1));

	for (int i = laplacian_pyramid.size() - 2; i >= 0; i--)
		reconstructions.push_back(laplacian_pyramid.at(i) + expand(reconstructions.at(levels_num-2-i), laplacian_pyramid.at(i).rows, laplacian_pyramid.at(i).cols));

	/*for (int i = 0; i < 3; i++) {
		laplacian_pyramid.at(i).convertTo(laplacian_pyramid.at(i), CV_8U);
		imshow("LP"+to_string(i), laplacian_pyramid.at(i));

	}*/

	/*
	for (int i = 3; i < 6; i++) {
		reconstructions.at(i).convertTo(reconstructions.at(i), CV_8U);
		imshow("R"+to_string(i), reconstructions.at(i));
	}*/


	return reconstructions.at(reconstructions.size()-1);



}

Mat BurtAdelsonGray(Mat imageA, Mat imageB, Mat mask){
	vector<Mat> laplacian_pyramidA = computeLaplacianPyramid(imageA);
	vector<Mat> laplacian_pyramidB = computeLaplacianPyramid(imageB);
	vector<Mat> mask_gaussian_pyramid = computeGaussianPyramid(mask);

	vector<Mat> sol_laplacian_pyramid = combineLaplacianPyramids(laplacian_pyramidA, laplacian_pyramidB, mask_gaussian_pyramid);

	Mat solution = restoreImageFromLP(sol_laplacian_pyramid);

	return solution;
}


Mat BurtAdelson(Mat imageA, Mat imageB, Mat mask){
	Mat solution;
	Mat channelsA[3];
	Mat channelsB[3];
	Mat sol_channels[3];

	if (imageA.channels() == 1)
		solution = BurtAdelsonGray(imageA,imageB, mask);
	else if (imageA.channels() == 3){
		split(imageA, channelsA);
		split(imageB, channelsB);

		for (int i = 0; i < 3; i++)
			sol_channels[i] = BurtAdelsonGray(channelsA[i], channelsB[i], mask);
			merge(sol_channels, 3, solution);
	}
	else
		cout << "Numero de canales no valido" << endl;

	return solution;
}


void showIm(Mat im) {
	namedWindow("window", 1);
	imshow("window", im);
	//waitKey();
	//destroyWindow("window");
}

Mat curvar_cilindro(Mat im, double f, double s){
	Mat imagen_curvada = Mat::zeros(im.rows, im.cols, CV_8U);

	int centro_x = im.cols/2;
	int centro_y = im.rows/2;

	for(int i = 0; i < im.rows; i++)
		for(int j = 0; j < im.cols; j++)
			imagen_curvada.at<uchar>(floor(s*((i-centro_y)/sqrt((j-centro_x)*(j-centro_x)+f*f)) + centro_y),
								floor(s*atan((j-centro_x)/f) + centro_x) ) = im.at<uchar>(i,j);

	return imagen_curvada;
}

Mat curvar_esfera(Mat im, double f, double s){
	Mat imagen_curvada = Mat::zeros(im.rows, im.cols, CV_8U);

	int centro_x = im.cols/2;
	int centro_y = im.rows/2;

	for(int i = 0; i < im.rows; i++)
		for(int j = 0; j < im.cols; j++)
			imagen_curvada.at<uchar>(floor(s*atan((i-centro_y)/sqrt((j-centro_x)*(j-centro_x)+f*f)) + centro_y),
								floor(s*atan((j-centro_x)/f) + centro_x) ) = im.at<uchar>(i,j);

	return imagen_curvada;
}

//Buqueda de traslacion


float distance(Mat im1, Mat im2, int t){
	float distance = 0;
	int numPixelUsed = 0;

	for(int col = t; col < im1.cols; col++)
		for(int row = 0; row < im1.rows; row++)
			if(im1.at<uchar>(row,col) != 0 && im2.at<uchar>(row,col) != 0){
				distance += abs(im1.at<uchar>(row,col) - im2.at<uchar>(row,col));
				numPixelUsed++;
			}

	//if(numPixelUsed < pequeño)
	// 	return ERROR //por considerar muy muy pocos puntos

	return distance/numPixelUsed;
}

int getTraslation1C(Mat &im1, Mat &im2){
	float min = 1000;
	int traslation = -1;
	float current_distance;

	for(int t = 0; t < im1.cols; t++){
		current_distance = distance(im1,im2,t);
		if (current_distance < min){
			min = current_distance;
			traslation = t;
		}
	}

	return traslation;
}

int getTraslation(Mat &im1, Mat &im2) {
	Mat im1_channels[3];
	Mat im2_channels[3];
	int traslation = 0;

	if (im1.channels() == 1)
		traslation = getTraslation1C(im1, im2);
	else if (im1.channels() == 3) {
		split(im1, im1_channels);
		split(im2, im2_channels);

		for (int i = 0; i < 3; i++)
			traslation += getTraslation1C(im1_channels[i], im2_channels[i]);

		traslation /= 3; //no estoy tomando decimales
	}
	else
		cout << "Numero de canales no valido" << endl;

	return traslation;
}


/*
Funcion que hace un mosaico con dos imagenes con un solo canal
@im1: una de las imagenes que forman el mosaico
@im2: la otra imagen para formar el mosaico

Mat makeMosaic1C (Mat im1, Mat im2) {
	int traslation = getTraslation(im1, im2);

	Mat expanded_im1 = Mat::zeros(im1.rows, im1.cols + traslation, im1.type());
	Mat expanded_im2 = Mat::zeros(im1.rows, im1.cols + traslation, im1.type());

	//Expandimos las imagenes
	for (int r = 0; r < im1.rows; r++)
		for (int c = 0; c < im1.cols; c++) {
			expanded_im1.at<float>(r,c) = im1.at<float>(r,c);
			expanded_im2.at<float>(r, c+traslation) = im2.at<float>(r,c);
		}

	//Creamos una mascara para B-A
	Mat mask = Mat::zeros(expanded_im1.rows, expanded_im1.cols, CV_32F);

	for (int r = 0; i < mask.rows; i++)
		for (int c = 0; c < mask.cols; c++)
			if (expanded_im1.at<float>(r,c) != 0.0 || expanded_im2.at<float>(r,c) != 0.0)
				mask.at<float>(r,c) = 1.0;

	return BurtAdelsonGray(expanded_im1, expanded_im2, mask);
}
*/

/*
Funcion que hace un mosaico con dos imagenes
@im1: una de las imagenes que forman el mosaico
@im2: la otra imagen para formar el mosaico
*/
Mat makeMosaic (Mat im1, Mat im2) {
	//int traslation = getTraslation(im1, im2);
	int traslation = 100;

	Mat expanded_im1 = Mat::zeros(im1.rows, im1.cols + traslation, im1.type());
	Mat expanded_im2 = Mat::zeros(im1.rows, im1.cols + traslation, im1.type());
	Mat mask = Mat::zeros(expanded_im1.rows, expanded_im1.cols, CV_32F);
	
	cout << "Las expandidas tienen: " << expanded_im1.rows << " filas y " << expanded_im1.cols << " cols." << endl;
	
	Mat expanded_im1_ROI = expanded_im1(Rect(0,0,im1.rows, im1.cols));
	Mat expanded_im2_ROI = expanded_im2(Rect(traslation-1,0, im2.rows, im2.cols));

	im1.copyTo(expanded_im1_ROI);
	im2.copyTo(expanded_im2_ROI);

	Mat expanded_im1_gray;
	cout << "Voy a convertir el color: " << endl;
	if (im1.channels() == 3)
		cvtColor(expanded_im1, expanded_im1_gray, CV_RGB2GRAY);
	else
		expanded_im1.copyTo(expanded_im1_gray);
	
	cout << "Canales de expanded_im1_gray: " << expanded_im1_gray.channels() << endl;
	
	Mat exp;
	expanded_im1_gray.convertTo(exp, CV_8U);
	imshow("La expandida de la manzana en gris", exp);

	for (int r = 0; r < mask.rows; r++)
		for (int c = 0; c < mask.cols; c++)
			if (expanded_im1_gray.at<float>(r,c) != 0.0)
				mask.at<float>(r,c) = 1.0;
				
	Mat mask2 = Mat(mask.rows, mask.cols, mask.type());
	
	for (int r = 0; r < mask.rows; r++)
		for (int c = 0; c < mask.cols; c++)
			mask2.at<float>(r,c) = 255 *mask.at<float>(r,c);
			
	Mat mosaic = BurtAdelson(expanded_im1, expanded_im2, mask);
	
	Mat una, otra, m;
	expanded_im1.convertTo(una, CV_8UC3);
	imshow("Expandida de la manzana", una);
	expanded_im2.convertTo(otra, CV_8UC3);
	imshow("Expandidad naranja", otra);
	mask2.convertTo(m, CV_8U);
	imshow("La mascara usada", m);
	
	return mosaic;
}

int main(int argc, char* argv[]){
	//EJEMPLO PARA PROBAR MOSAICO
	Mat apple = imread("imagenes/apple.jpeg");
	Mat orange = imread("imagenes/orange.jpeg");
	
	orange.convertTo(orange, CV_32FC3);
	apple.convertTo(apple, CV_32FC3);
	
	Mat mosaic = makeMosaic(apple, orange);
	mosaic.convertTo(mosaic, CV_8UC3);
	imshow("El mosaico", mosaic);
	
	//EJEMPLO PARA PROBAR TRASLACION
	
	/*Mat mosaic1 = imread("imagenes/mosaic1.png", 0);
	Mat mosaic2 = imread("imagenes/mosaic2.png", 0);

	//imshow("Mosaic1", mosaic1);
	//imshow("Mosaic2", mosaic2);
	cout << "Las columnas de mosaic1 son: " << mosaic1.cols << endl;
	cout << "La traslacion es: " << getTraslation(mosaic1, mosaic2) << endl;*/


	//EJEMPLO PARA VER CILINDRICAS Y ESFERICAS

	/*Mat imagen_cilindro, imagen_esfera;
	imagen_cilindro = curvar_cilindro(imagen,500,500);
	imagen_esfera = curvar_esfera(imagen,500,500);


	imshow("Normal", imagen);
	imshow("Imagen cilindro", imagen_cilindro);
	imshow("Imagen esfera", imagen_esfera);*/

	//EJEMPLO PARA VER SI VA BIEN LAS RECONSTRUCCION O NO
	/*Mat image = imread("imagenes/Image1.tif", 0);
	image.convertTo(image, CV_32F);
	vector<Mat> laplacianPyramid = computeLaplacianPyramid(image);
	Mat reconstruction = restoreImageFromLP(laplacianPyramid);
	reconstruction.convertTo(reconstruction, CV_8U);
	imshow("Reconstruccion tablero",reconstruction);*/



	//EJEMPLO PARA PROBAR B-A EN COLOR

	/*Mat apple = imread("imagenes/apple.jpeg");
	Mat orange = imread("imagenes/orange.jpeg");
	Mat mask = imread("imagenes/mask_apple_orange.png", 0);

	imshow("apple.jpeg", apple);
	imshow("oragne.jpeg", orange);
	imshow("mask_apple_orange.png", mask);

	apple.convertTo(apple, CV_32FC3);
	orange.convertTo(orange, CV_32FC3);
	mask.convertTo(mask, CV_32F);

	Mat current_mask = Mat(mask.rows, mask.cols, CV_32F);

	for (int i = 0; i < current_mask.rows; i++)
		for (int j = 0; j < current_mask.cols; j++)
			if (mask.at<float>(i,j) < 127)
				current_mask.at<float>(i,j) = 0.0;
			else
				current_mask.at<float>(i,j) = 1.0;

	Mat combination = BurtAdelson(orange, apple, current_mask);

	combination.convertTo(combination, CV_8UC3);

	imshow("combination", combination);*/

	waitKey();
	destroyAllWindows();

    return 0;
}
