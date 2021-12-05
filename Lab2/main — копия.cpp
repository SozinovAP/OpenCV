#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <math.h>

//! 1. Convolution with 5 filters 3x3x3 (random numbers)
//! 2. Normalize (coefs. and offset - arbitrary)
//! 3. ReLU
//! 4. MAX POOLING (2x2)
//! 5. SoftMax for each pixel

using namespace cv;

typedef std::array<std::array<std::array<int, 3>, 3>, 3> Matrix;
namespace
{
  // count filter for convolution
  int FILTERS_COUNT = 5;
  int SIZE_MATRIX = 3;

  Mat Convolution(const Mat& theImage, const std::vector<Matrix>& theFilters)
  {
    auto aCols = theImage.cols;
    auto aRows = theImage.rows;
    int sizes[] = { FILTERS_COUNT, aRows, aCols };
    Mat aRes = Mat::zeros(3, sizes, theImage.type());

    for (int filtInd = 0; filtInd < FILTERS_COUNT; ++filtInd)
      aRes.forEach<Point3_<uint8_t>>(
        [&](Point3_<uint8_t>& pixel, const int* position)
        {
          for (int dimX = 0; dimX < SIZE_MATRIX; ++dimX)
            for (int dimY = 0; dimY < SIZE_MATRIX; ++dimY)
            {
              pixel.x += theImage.at<Point3_<uint8_t>>(
                std::min(position[0] + dimX, aRows - 1), std::min(position[1] + dimY, aCols - 1)).x
                * theFilters[filtInd][0][dimX][dimY];
              pixel.y += theImage.at<Point3_<uint8_t>>(
                std::min(position[0] + dimX, aRows - 1), std::min(position[1] + dimY, aCols - 1)).y
                * theFilters[filtInd][1][dimX][dimY];
              pixel.z += theImage.at<Point3_<uint8_t>>(
                std::min(position[0] + dimX, aRows - 1), std::min(position[1] + dimY, aCols - 1)).z
                * theFilters[filtInd][2][dimX][dimY];
            }
        });
    //for (int filtInd = 0; filtInd < FILTERS_COUNT; ++filtInd)
    //  for (int aCol = 0; aCol < aCols; ++aCol)
    //    for (int aRow = 0; aRow < aRows; ++aRow)
    //      for (int dimX = 0; dimX < SIZE_MATRIX; ++dimX)
    //        for (int dimY = 0; dimY < SIZE_MATRIX; ++dimY)
    //          for (int dimZ = 0; dimZ < SIZE_MATRIX; ++dimZ)
    //          {
    //            aRes.at<Point3_<uint8_t>>(/*filtInd,*/ aRow, aCol) +=
    //              theImage.at<Point3_<uint8_t>>(/*dimZ,*/ std::min(aRow + dimX, aRows - 1),
    //                std::min(aCol + dimY, aCols - 1)) * theFilters[filtInd][dimZ][dimX][dimY];
    //          }
    return aRes;
  }

  Mat Normalize(const Mat& theImage)
  {
    Mat aRes = theImage;

    return aRes;
  }

  Mat ReLU(const Mat& theImage)
  {
    Mat aRes = theImage;

    return aRes;
  }

  Mat Pooling(const Mat& theImage)
  {
    Mat aRes = theImage;

    return aRes;
  }

  Mat SoftMax(const Mat& theImage)
  {
    //    cv::sum
    Mat aRes = theImage;
    aRes.forEach<Point3_<uint8_t>>(
      [](Point3_<uint8_t>& pixel, const int* position) -> void
      {
        auto aSum = cv::sum(std::vector<uint8_t>{ pixel.x, pixel.y, pixel.z });
      }
    );
    //if (pow(double(pixel.x) / 10, 2.5) > 100)
    //{
    //  pixel.x = 100;
    //  pixel.y = 100;
    //  pixel.z = 255;
    //}
    //else
    //{
    //  pixel.x = 0;
    //  pixel.y = 0;
    //  pixel.z = 0;
    //}
    return aRes;
  }

  void GenerateNumbers(std::vector<Matrix>& theFilters)
  {
    std::random_device randomDevice;
    std::mt19937 mersEngine(randomDevice());
    std::uniform_real_distribution<float> dist(1, 10);
    auto gener = [&]()
    {
      Matrix res;
      for (auto& resX : res)
        for (auto& resY : resX)
          for (auto& resZ : resY)
            resZ = dist(mersEngine);
      return res;
    };

    std::generate(theFilters.begin(), theFilters.end(), gener);


  }
}

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
  Mat image{ imread("./res/cat.jpg") };
  if (showImage(image, "Display Window") == -1)
    return -1;

  // Convolution
  std::vector<Matrix> aFilters(FILTERS_COUNT);
  GenerateNumbers(aFilters);
  auto resConv = Convolution(image, aFilters);

  // Normalize
  auto aNormImg = Normalize(resConv);

  // ReLU
  auto aReLUImg = ReLU(aNormImg);

  // Pooling 2x2
  auto aPoolingImg = Pooling(aReLUImg);

  // Softmax
  auto aSoftMax = SoftMax(aPoolingImg);

  std::vector<Mat> matVec;
  std::string aTitle = "Result Image";
  for (int anIndex = 0; anIndex < FILTERS_COUNT; ++anIndex)
  {
    matVec.push_back(cv::Mat(image.rows, image.cols, image.type(),
      aSoftMax.data + anIndex * image.rows * image.cols).clone());
    if (showImage(matVec.back(), aTitle + std::to_string(anIndex)) == -1)
      return -1;
  }


  //Mat aCol = Mat::zeros(image.size(), image.type());
  //aCol.forEach<Point3_<uint8_t>>(
  //  [&matVec](Point3_<uint8_t>& pixel, const int* position) -> void
  //  {
  //    for (const auto& aMat : matVec)
  //    {
  //      pixel += aMat.at< Point3_<uint8_t>>(position[0], position[1]);
  //    }
  //  });
  //if (showImage(aCol, "Hoba") == -1)
  //  return -1;


  waitKey(0);
  return 0;
}