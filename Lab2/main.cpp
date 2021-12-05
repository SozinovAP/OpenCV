#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <math.h>
#include <mutex>
#include <execution>

//! 1. Convolution with 5 filters 3x3x3 (random numbers)++
//! 2. Normalize (coefs. and offset - arbitrary)
//! 3. ReLU++
//! 4. MAX POOLING (2x2)++
//! 5. SoftMax for each pixel

using namespace cv;

typedef std::array<std::array<std::array<float, 3>, 3>, 3> Matrix;
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
    Mat aRes = Mat::zeros(3, sizes, CV_32F);

    aRes.forEach<float>([&](float& theValue, const int* thePosition)
      {
        float sum = theValue;
        for (int i = 0; i < SIZE_MATRIX; ++i)
          for (int j = 0; j < SIZE_MATRIX; ++j)
          {
            auto xx = theImage.at<Point3_<uint8_t>>(std::min(thePosition[1] + i, aRows - 1),
              std::min(thePosition[2] + j, aCols - 1));
            sum += xx.x * theFilters[thePosition[0]][0][i][j];
            sum += xx.y * theFilters[thePosition[0]][1][i][j];
            sum += xx.z * theFilters[thePosition[0]][2][i][j];
          }
        theValue = sum;
      });
    /* for (int filtInd = 0; filtInd < FILTERS_COUNT; ++filtInd)
       for (int aRow = 0; aRow < aRows; ++aRow)
         for (int aCol = 0; aCol < aCols; ++aCol)
         {
           Point3f sum = { 0, 0, 0 };
           auto a = aRes.at<Point3f>(filtInd, aRow, aCol);
           for (int i = 0; i < SIZE_MATRIX; ++i)
             for (int j = 0; j < SIZE_MATRIX; ++j)
             {
               auto xx = theImage.at<Point3f>(std::min(aRow + i, aRows - 1),
                 std::min(aCol + j, aCols - 1)) / 255;
               sum.x += xx.x * theFilters[filtInd][0][i][j];
               sum.y += xx.y * theFilters[filtInd][1][i][j];
               sum.z += xx.z * theFilters[filtInd][2][i][j];
             }
           aRes.at<Point3f>(filtInd, aRow, aCol) = sum;
         }*/

    return aRes;
  }

  Mat Normalize(const Mat& theImage, const float theGamma = 1.f, const float& theBetta = 0.f)
  {
    Mat aRes;

    cv::normalize(theImage, aRes);
    return aRes;
  }

  Mat ReLU(const Mat& theImage)
  {
    Mat aRes = theImage;
    aRes.forEach<float>([&](float& pixel, const int* position) -> void
      {
        pixel = std::max(0.f, pixel);
      });
    return aRes;
  }

  Mat Pooling(const Mat& theImage, int thePoolSize = 2)
  {
    auto aCols = theImage.cols / thePoolSize;
    auto aRows = theImage.rows / thePoolSize;

    Mat aRes = Mat::zeros({ aCols, aRows }, CV_32F);

    aRes.forEach<float>([&](float& thePixel, const int* position)
      {
        float aPixelMax;
        for (int x = position[1] * thePoolSize; x < (position[1] + 1) * thePoolSize; ++x)
          for (int y = position[0] * thePoolSize; y < (position[0] + 1) * thePoolSize; ++y)
          {
            float aPixel = theImage.at<float>(y, x);
            aPixelMax = std::max(aPixelMax, aPixel);
          }
        thePixel = aPixelMax;
      });

    return aRes;
  }

  Mat SoftMax(const Mat& theImage)
  {
    auto aCols = theImage.cols;
    auto aRows = theImage.rows;

    Mat aRes = Mat::zeros({ aCols, aRows }, CV_32F);
    Mat anExpRes = Mat::zeros({ aCols, aRows }, CV_32F);
    exp(theImage, anExpRes);

    float aSumExp = 0;
    theImage.forEach<float>([&aSumExp](float& pixel, const int* position) -> void
      {
        aSumExp += exp(pixel);
      });

    aRes.forEach<float>(
      [&aSumExp, &anExpRes](float& pixel, const int* position) -> void
      {
        auto y = anExpRes.at<float>(position[0], position[1]);
        auto aValues = y / aSumExp;
        pixel = aValues;
      }
    );

    return aRes;
  }

  void GenerateNumbers(std::vector<Matrix>& theFilters)
  {
    std::random_device randomDevice;
    std::mt19937 mersEngine(randomDevice());
    std::uniform_real_distribution<float> dist(0, 1);
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
  Mat image{ imread("./res/dog.jpg") };
  if (showImage(image, "Display Window") == -1)
    return -1;

  // Convolution
  std::vector<Matrix> aFilters(FILTERS_COUNT);
  GenerateNumbers(aFilters);
  auto resConv = Convolution(image, aFilters);
  std::vector<Mat> aLayouts;

  for (int anIndex = 0; anIndex < FILTERS_COUNT; ++anIndex)
  {
    aLayouts.push_back(cv::Mat(image.rows, image.cols, CV_32F,
      resConv.data + anIndex * image.rows * image.cols).clone());
  }

  std::for_each(std::execution::par, aLayouts.begin(), aLayouts.end(), [&](auto& aMat)
    {
      // Normalize
      auto aNormImg = Normalize(aMat);
      //std::cout << aNormImg;

      // ReLU
      auto aReLUImg = ReLU(aNormImg);
      //std::cout << aReLUImg;

      // Pooling 2x2
      auto aPoolingImg = Pooling(aReLUImg);
      //std::cout << aPoolingImg;

      // Softmax
      auto aSoftMax = SoftMax(aPoolingImg);
      aMat = aSoftMax;
    });

  for (int anIndex = 0; anIndex < FILTERS_COUNT; ++anIndex)
  {
    std::cout << aLayouts[anIndex];
  }

  waitKey(0);
  return 0;
}