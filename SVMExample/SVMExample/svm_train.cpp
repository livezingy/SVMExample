/* --------------------------------------------------------
* 作者：livezingy
*
* 博客：http://www.livezingy.com
*
* 开发环境：
*      Visual Studio V2013
opencv3.1
*
* 版本历史：
*      V1.0    2017年1月30日
可使用LBP特征,HOG特征,SIFT/SURF特征实现SVM训练
--------------------------------------------------------- */

#include "stdafx.h"
#include "svm_train.h"
#include "util.h"
#include "lbp.hpp"
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/ml/ml.hpp>  
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <xfeatures2d/nonfree.hpp>
#include "BagOfWords.h"
#ifdef OS_WINDOWS
#include <ctime>
#endif

using namespace std;
using namespace cv;
using namespace cv::ml;

extern string resPath;

SvmTrain::SvmTrain()
{
  pBOW = BagOfWords();
}

void SvmTrain::trainAuto()
{
	string tempPath = resPath + feaType;
	utils::mkdir(tempPath);
	/* first check if a previously trained svm for the current class has been saved to file */
	string svmFilename = tempPath + "/" + feaType + ".xml.gz";

	FileStorage fs(svmFilename, FileStorage::READ);
	if (fs.isOpened())
	{
		cout << "*** LOADING SVM CLASSIFIER***" << endl;
		svm_ = StatModel::load<SVM>(svmFilename);
	}
	else
	{
		svm_ = cv::ml::SVM::create();
		svm_->setType(cv::ml::SVM::C_SVC);
		svm_->setKernel(cv::ml::SVM::KernelTypes::RBF);
		svm_->setTermCriteria(cvTermCriteria(CV_TERMCRIT_ITER, 40000, 0.00001));
		auto train_data = tdata();
		svm_->trainAuto(train_data, 10, SVM::getDefaultGrid(SVM::C),
			SVM::getDefaultGrid(SVM::GAMMA), SVM::getDefaultGrid(SVM::P),
			SVM::getDefaultGrid(SVM::NU), SVM::getDefaultGrid(SVM::COEF),
			SVM::getDefaultGrid(SVM::DEGREE), true);

		fprintf(stdout, ">> Saving model file...\n");
		
		svm_->save(svmFilename);

		fprintf(stdout, ">> Your SVM Model was saved to %s\n", svmFilename);
		fprintf(stdout, ">> Testing...\n");		
	}
	
	this->test();
}


void SvmTrain::train() 
{
	string tempPath = resPath + feaType;
	utils::mkdir(tempPath);
	
	string svmFilename = resPath + feaType + "/" + feaType + ".xml.gz";

	FileStorage fs(svmFilename, FileStorage::READ);
	if (fs.isOpened())
	{
		cout << "*** LOADING SVM CLASSIFIER***" << endl;
		svm_ = StatModel::load<SVM>(svmFilename);
	}
	else
	{
		svm_ = cv::ml::SVM::create();
		svm_->setType(cv::ml::SVM::Types::C_SVC);
		svm_->setKernel(cv::ml::SVM::KernelTypes::RBF);
		svm_->setDegree(0);
		svm_->setGamma(1); 
		svm_->setCoef0(0);
		svm_->setC(0.1);
		svm_->setNu(0);
		svm_->setP(0);
		svm_->setTermCriteria(cvTermCriteria(CV_TERMCRIT_ITER, 10000, 0.0001));
		/*
		typedef struct CvTermCriteria
		{
		//【1】int type--type of the termination criteria,one of:
		//【1】int type---迭代算法终止条件的类型，是下面之一:
		//【1】CV_TERMCRIT_ITER---在完成最大的迭代次数之后,停止算法
		//【2】CV_TERMCRIT_EPS----当算法的精确度小于参数double epsilon指定的精确度时，停止算法
		//【3】CV_TERMCRIT_ITER+CV_TERMCRIT_EPS--无论是哪一个条件先达到,都停止算法
		int    type;   may be combination of
		CV_TERMCRIT_ITER
		CV_TERMCRIT_EPS 
		//【2】Maximum number of iterations  
		//【2】最大的迭代次数  
		int    max_iter;
		//【3】Required accuracy  
		//【3】所要求的精确度  
		double epsilon;
		}
		*/

		auto train_data = tdata();

		fprintf(stdout, ">> Training SVM model, please wait...\n");
		long start = utils::getTimestamp();

		svm_->train(train_data);

		long end = utils::getTimestamp();
		fprintf(stdout, ">> Training done. Time elapse: %ldms\n", end - start);
		fprintf(stdout, ">> Saving model file...\n");
		svm_->save(svmFilename);

		fprintf(stdout, ">> Your SVM Model was saved to %s\n", svmFilename);
		fprintf(stdout, ">> Testing...\n");
	}
	
    this->test();
}

void SvmTrain::test() 
{
  if (test_file_list_.empty())
  {
      this->prepare();
  }

  double count_all = test_file_list_.size();
  double ptrue_rtrue = 0;
  double ptrue_rfalse = 0;
  double pfalse_rtrue = 0;
  double pfalse_rfalse = 0;

  for (auto item : test_file_list_)
  {
    auto image = cv::imread(item.file);
    if (!image.data)
	{
      
      std::cout << "no" << std::endl;
      continue;
    }

    cv::Mat feature;
	
    if(feaType == "LBP")
	{
		getLBPFeatures(image, feature);
	}
	else
	{
		getHOGFeatures(image, feature);
	}

    auto predict = int(svm_->predict(feature));
    std::cout << "predict: " << predict << std::endl;

    auto real = item.label;
    if (predict == kForward && real == kForward) ptrue_rtrue++;
    if (predict == kForward && real == kInverse) ptrue_rfalse++;
    if (predict == kInverse && real == kForward) pfalse_rtrue++;
    if (predict == kInverse && real == kInverse) pfalse_rfalse++;
  }

  std::cout << "count_all: " << count_all << std::endl;
  std::cout << "ptrue_rtrue: " << ptrue_rtrue << std::endl;
  std::cout << "ptrue_rfalse: " << ptrue_rfalse << std::endl;
  std::cout << "pfalse_rtrue: " << pfalse_rtrue << std::endl;
  std::cout << "pfalse_rfalse: " << pfalse_rfalse << std::endl;

  double precise = 0;
  if (ptrue_rtrue + ptrue_rfalse != 0) 
  {
    precise = ptrue_rtrue / (ptrue_rtrue + ptrue_rfalse);
    std::cout << "precise: " << precise << std::endl;
  } 
  else 
  {
    std::cout << "precise: "
              << "NA" << std::endl;
  }

  double recall = 0;
  if (ptrue_rtrue + pfalse_rtrue != 0) {
    recall = ptrue_rtrue / (ptrue_rtrue + pfalse_rtrue);
    std::cout << "recall: " << recall << std::endl;
  } else {
    std::cout << "recall: "
              << "NA" << std::endl;
  }

  double Fscore = 0;
  if (precise + recall != 0) {
	  Fscore = 2 * (precise * recall) / (precise + recall);
	  std::cout << "Fscore: " << Fscore << std::endl;
  } else {
    std::cout << "Fscore: "
              << "NA" << std::endl;
  }

}

void SvmTrain::getLBPFeatures(const cv::Mat& image, cv::Mat& features) 
{

	Mat grayImage;
	cvtColor(image, grayImage, CV_RGB2GRAY);

	Mat lbpimage;
	lbpimage = libfacerec::olbp(grayImage);

	//spatial_histogram函数返回的特征至为单行且数据类型为CV_32FC1
	Mat lbp_hist = libfacerec::spatial_histogram(lbpimage, 32, 4, 4);

	features = lbp_hist;
}

void SvmTrain::getBOWFeatures()
{
	vector<string> imgPathTrain;
	
	vector<string> tmpImgTrain;

	char buffer[260] = { 0 };

	sprintf(buffer, "DATA/has/train");
	imgPathTrain = utils::getFiles(buffer);
	vector<char> objPresentTrain(imgPathTrain.size(), 1);

	sprintf(buffer, "DATA/no/train");
	tmpImgTrain = utils::getFiles(buffer);

	for (auto file : tmpImgTrain)
	{
		imgPathTrain.push_back(file);
		objPresentTrain.push_back(0);
	}

	

	vector<string> imgPathTest;

	vector<string> tmpImgTest;

	sprintf(buffer, "DATA/has/test");
	imgPathTest = utils::getFiles(buffer);
	vector<char> objPresentTest(imgPathTest.size(), 1);

	sprintf(buffer, "DATA/no/test");
	tmpImgTest = utils::getFiles(buffer);

	for (auto file : tmpImgTest)
	{
		imgPathTest.push_back(file);
		objPresentTest.push_back(0);
	}

	pBOW.getVocabulary(imgPathTrain, objPresentTrain, imgPathTest, objPresentTest);
}

void SvmTrain::getHOGFeatures(const cv::Mat& image, cv::Mat& features)
{
	//此参数下特征维度为2340，LINER和RBF内核均trainAuto均可得到结果
	HOGDescriptor hog;
	hog.winSize = cvSize(image.cols, image.rows);// Size();//136*36
	hog.blockSize = Size(40, 4);
	hog.blockStride = Size(8, 8);
	hog.cellSize = Size(20, 2);

	
	/*
	//此参数下得到的特征维度为24156
	//此参数下RBF内核trainAuto时无法得出结果
	HOGDescriptor hog(cvSize(136, 36), cvSize(16, 16), cvSize(8, 8), cvSize(8, 8), 9);
	hog.blockStride = Size(2, 2);
	*/

	vector<float> descriptors;

	Mat gray;

	cvtColor(image, gray, COLOR_BGR2GRAY);

	hog.compute(gray, descriptors, Size(2, 2), Size(0, 0));

	cv::Mat tmp(1, Mat(descriptors).cols, CV_32FC1); //< used for transposition if needed

	//hog计算得到的特征cols=1,rows=维数
	transpose(Mat(descriptors), tmp);

	tmp.copyTo(features);

}


void SvmTrain::prepare() {
	
  srand(unsigned(time(NULL)));

  char buffer[260] = {0};

  sprintf(buffer, "DATA/has/train");
  auto has_file_train_list = utils::getFiles(buffer);
  std::random_shuffle(has_file_train_list.begin(), has_file_train_list.end());

  sprintf(buffer, "DATA/has/test");
  auto has_file_test_list = utils::getFiles(buffer);
  std::random_shuffle(has_file_test_list.begin(), has_file_test_list.end());

  sprintf(buffer, "DATA/no/train");
  auto no_file_train_list = utils::getFiles(buffer);
  std::random_shuffle(no_file_train_list.begin(), no_file_train_list.end());

  sprintf(buffer, "DATA/no/test");
  auto no_file_test_list = utils::getFiles(buffer);
  std::random_shuffle(no_file_test_list.begin(), no_file_test_list.end());

  fprintf(stdout, ">> Collecting train data...\n");

  for (auto file : has_file_train_list)
    train_file_list_.push_back({ file, kForward });

  for (auto file : no_file_train_list)
    train_file_list_.push_back({ file, kInverse });

  fprintf(stdout, ">> Collecting test data...\n");

  for (auto file : has_file_test_list)
    test_file_list_.push_back({ file, kForward });

  for (auto file : no_file_test_list)
    test_file_list_.push_back({ file, kInverse });
}

cv::Ptr<cv::ml::TrainData> SvmTrain::tdata() {
  this->prepare();
        
  cv::Mat samples;
  std::vector<int> responses;

	  for (auto f : train_file_list_) {
		  auto image = cv::imread(f.file);
		if (!image.data) {
		  fprintf(stdout, ">> Invalid image: %s  ignore.\n", f.file.c_str());
		  continue;
		}
		cv::Mat feature;

		if(feaType == "LBP")
		{
			getLBPFeatures(image, feature);
		}
		else
		{
			getHOGFeatures(image, feature);
		}

		samples.push_back(feature);
		responses.push_back(int(f.label));
	  }

	  //create：Creates training data from in-memory arrays.
	  //samples_：matrix of samples. It should have CV_32F type
	  //ROW_SAMPLE:each training sample is a row of samples
	  //responses_：matrix of responses. If the responses are scalar, 
	  //they should be stored as a single row or as a single column. The matrix should have type CV_32F or CV_32S
	  return cv::ml::TrainData::create(samples, cv::ml::SampleTypes::ROW_SAMPLE,
		  Mat(responses));
}


