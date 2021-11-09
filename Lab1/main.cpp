#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>

// 1. Find face on image
// 2. cut out a fragment, departing 10% from the borders of the face
// 3. Get a binary image of object boundaries
// 4. Remove boundaries with Length and Height are less than 10
// 5. Apply morphological augmentation operation
// 6. Smooth the resulting image of the edges with Gaussian filter (5x5). 
// Get normalized image M, where all pixels are from 0 to 1
// 7. Get image F1 of a face with bilateral filtering applied
// 8. Get image F2 of a face with improved contrast/clarity
// 9. Final filtering according to the formula:
// Result[x,y] = M[x,y] * F2[x,y] + (1 - M[x,y]) * F1[x,y]

using namespace cv;

int main(int argc, char** argv)
{
	auto showImage = [&](const Mat& theImage, const String& theTitle = "")
	{
		if (theImage.empty())
		{
			std::cerr << "Image is empty!!";
			return -1;
		}
		imshow(theTitle.empty() ? "Image" : theTitle, theImage);
		return 0;
	};
	// Load image
	Mat image{ imread("./res/img.png") };
	if (showImage(image, "Display Window") == -1)
		return -1;

	// step 1 - Find face
	std::vector<Rect> detectedFaces;
	auto faceCascade{ CascadeClassifier("./res/haarcascade_frontalface_alt.xml") };
	faceCascade.detectMultiScale(image, detectedFaces);

	if (detectedFaces.empty())
	{
		std::cerr << "Faces is not found!";
		return -1;
	}
	auto face{ detectedFaces.front() };
	Mat faceImg{ image(face) };
	if(showImage(faceImg, "Face") == -1)
		return -1;

	// step 2 - cut fragment
	Rect extFaceRect{ face };
	int offsetX = static_cast<int>(face.width / 10);
	int offsetY = static_cast<int>(face.height / 10);

	extFaceRect.width += offsetX;
	extFaceRect.height += offsetY;
	extFaceRect.x -= offsetX;
	extFaceRect.y -= offsetY;

	Mat extFaceImg{ image(extFaceRect) };
	if(showImage(extFaceImg, "Face") == -1)
		return -1;

	// step 3 - get binary image
	Mat binImg;
	Canny(extFaceImg, binImg, 30, 140); // Try use another value
	if(showImage(binImg, "Bin") == -1)
		return -1;

	// step 4 - find and remove small contours
	std::vector<std::vector<Point>> contours, newContours;
	cv::findContours(binImg, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
	Mat removeEdges{ Mat::zeros(binImg.size(), binImg.type()) };

	for (const auto& contour : contours)
	{
		Rect boundRect{ boundingRect(contour) };
		if (boundRect.width >= 10 && boundRect.height >= 10)
		{
			newContours.push_back(contour);
		}
	}
	drawContours(removeEdges, newContours, -1, { 255, 255, 255 }, 1);
	if(showImage(removeEdges, "Remove small edges") == -1)
		return -1;

	//Mat edgeLarge;
	//Mat mask = Mat::zeros(binImg.size(), binImg.type());

	//// drawContours(mask, newContours, -1
	//drawContours(mask, newContours, 0, { 255, 255, 255 }, 1);
	//bitwise_and(binImg, binImg, edgeLarge, mask);
	//showImage(edgeLarge, "Remove");

	//Mat res;
	// bitwise_and(binImg, binImg, edgeLarge, mask);
	// showImage(edgeLarge, "Remove");

	// step 5 - morphological operation
	Mat structElem{ getStructuringElement(cv::MORPH_RECT, { 5,5 }) };
	Mat morphOperatImg;
	dilate(removeEdges, morphOperatImg, structElem);
	if(showImage(morphOperatImg, "Morphological operation") == -1)
		return -1;

	// step 6 - gaussian image and normalize this
	Mat gaussianImg;
	GaussianBlur(morphOperatImg, gaussianImg, { 5,5 }, 0);
	if(showImage(gaussianImg, "Gaussian") == -1)
		return -1;

	Mat normalizeMImg;
	normalize(gaussianImg, normalizeMImg, 0, 1, cv::NORM_MINMAX);
	if(showImage(normalizeMImg * 255, "Normalize") == -1)
		return -1;

	// step 7 - bilateral
	Mat F1Img;
	bilateralFilter(extFaceImg, F1Img, 50, 100, 30);
  if(showImage(F1Img, "Bilateral") == -1)
		return -1;

	// step 8 - contrast
	Mat F2Img;
	Mat kernel{ (Mat_<double>(3, 3) << -1, -1, -1, -1, 9, -1, -1, -1, -1) };
	filter2D(extFaceImg, F2Img, -1, kernel);
	if(showImage(F2Img, "Contrast") == -1)
		return -1;

	// step 9 - final filter
	Mat normalizedBGR, normalizedBGRInverted;
	cvtColor(normalizeMImg, normalizedBGR, COLOR_GRAY2RGB);
	cvtColor((1 - normalizeMImg), normalizedBGRInverted, COLOR_GRAY2RGB);

	Mat resImg{ normalizedBGR.mul(F2Img) + normalizedBGRInverted.mul(F1Img) };

	if(showImage(resImg, "RESULT") == -1)
		return -1;

	waitKey(0);
	return 0;
}