Mat Reduce(Mat Matrix){

}

vector<Mat> ComputeGaussianpyramid(Mat image){
	vector<Mat> gaussianPyramid;
	Mat actualLevelMatrix = image;

	while (5 <= actualLevelMatrix.cols && 5 <= actualLevelMatrix.rows){
		gaussianPyramid.push_back(actualLevelMatrix);
		actualLevelMatrix = Reduce(actualLevelMatrix);
	}

	return gaussianPyramid;
}

Mat Expand(Mat Matrix){

}

vector<Mat> ComputeLaplacianpyramid(Mat image){
	vector<Mat> solution, gaussianPyramid;
	Mat actualLevelMatrix;

	gaussianPyramid = ComputeGaussianpyramid(image);

	for (int i = 0; i < gaussianPyramid.size(); i++){
		if(i < gaussianPyramid.size() - 1)
			actualLevelMatrix = gaussianPyramid.at(i) - Expand(gaussianPyramid.at(i+1));
		else
			actualLevelMatrix = gaussianPyramid.at(i);

		solution.push_back(actualLevelMatrix);
	}

	return solution;
}

vector<Mat> CombineLaplacianPyramids(vector<Mat> laplacianPyramidA,
									 vector<Mat> laplacianPyramidB,
									 vector<Mat> gaussianPyramidMask){

	vector<Mat> combinedPyramids;
	Mat actualLevelMatrix;

	for (int k = 0; k < laplacianPyramidA.size(); k++){
		actualLevelMatrix = Mat::zeros(laplacianPyramidA.at(k).rows,laplacianPyramidA.at(k).cols, CV_COSA);

		for(int i = 0; i < laplacianPyramidA.at(k).rows; i++)
			for(int j = 0; j < laplacianPyramidA.at(k).cols; j++)
				actualLevelMatrix.at<cosa>(j,i) += laplacianPyramidA.at(k).at<cosa>(j,i) * gaussianPyramidMask.at(k).at<cosa>(j,i) +
												   (1-gaussianPyramidMask.at(k).at<cosa>(j,i)) * laplacianPyramidB.at(k).at<cosa>(j,i);

		combinedPyramids.push_back(actualLevelMatrix);
	}

	return combinedPyramids;
}

Mat BurtAdelsonGray(Mat imageA, Mat imageB, Mat mask){
	vector<Mat> laplacianPyramidA = ComputeLaplacianpyramid(imageA);
	vector<Mat> laplacianPyramidB = ComputeLaplacianpyramid(imageB);
	vector<Mat> gaussianPyramidMask = ComputeGaussianpyramid(mask);

	vector<Mat> laplacianPyramidSol = CombineLaplacianPyramids(laplacianPyramidA,
															   laplacianPyramidB,
														       gaussianPyramidMask);

	Mat solution = RestoreImageFromLP(laplacianPyramidSol);
	return solution;
}

Mat BurtAdelson(Mat imageA, Mat imageB, Mat mask){
	Mat solution;

	if (imageA.chanels == 1)
		solution = BurtAdelsonGray(imageA,imageB, mask);
	else if (imageA.chanels == 3){
		split

		for (int i = 0; i < 3; i++)
			splitedSolution.push_back(BurtAdelsonGray(splitedImageA.at(i),splitedImageB.at(i), mask));

		solution = merge;
	}

	return solution;
}

int main(){
	Mat imageA,imageB, mask;
	imageA = loadImage('PATH');
	imageB = loadImage('PATH');
	mask = loadImage('PATH');

	Mat mixedImage;
	mixedImage = BurtAdelson(imageA, imageB, mask);
	showIm(mixedImage);
}
