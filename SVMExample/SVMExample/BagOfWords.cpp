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


#include "opencv2/opencv_modules.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/ml.hpp"
#include "util.h"
#include "BagOfWords.h"

#include <fstream>
#include <iostream>
#include <memory>
#include <functional>

#if defined WIN32 || defined _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#undef min
#undef max
#include "sys/types.h"
#endif
#include <sys/stat.h>

#define DEBUG_DESC_PROGRESS

using namespace cv;
using namespace cv::xfeatures2d;
using namespace cv::ml;
using namespace std;

extern string resPath;
const string paramsFile = "params.xml";
const string vocabularyFile = "vocabulary.xml.gz";
const string bowImageDescriptorsDir = "/bowImageDescriptors";
const string svmsDir = "/svms";
//const string plotsDir = "/plots";


class ObdImage
{
public:
	ObdImage(string p_id, string p_path) : id(p_id), path(p_path) {}
	string id;
	string path;
};

//
// This part of the code was a little refactor
//
struct DDMParams
{
	DDMParams() : detectorType("SURF"), descriptorType("SURF"), matcherType("BruteForce") {}
	DDMParams(const string _detectorType, const string _descriptorType, const string& _matcherType) :
		detectorType(_detectorType), descriptorType(_descriptorType), matcherType(_matcherType){}
	void read(const FileNode& fn)
	{
		fn["detectorType"] >> detectorType;
		fn["descriptorType"] >> descriptorType;
		fn["matcherType"] >> matcherType;
	}
	void write(FileStorage& fs) const
	{
		fs << "detectorType" << detectorType;
		fs << "descriptorType" << descriptorType;
		fs << "matcherType" << matcherType;
	}
	void print() const
	{
		cout << "detectorType: " << detectorType << endl;
		cout << "descriptorType: " << descriptorType << endl;
		cout << "matcherType: " << matcherType << endl;
	}

	string detectorType;
	string descriptorType;
	string matcherType;
};

struct VocabTrainParams
{
	VocabTrainParams() : vocabSize(1000), memoryUse(200), descProportion(0.3f) {}
	VocabTrainParams(const string _trainObjClass, size_t _vocabSize, size_t _memoryUse, float _descProportion) :
		vocabSize((int)_vocabSize), memoryUse((int)_memoryUse), descProportion(_descProportion) {}
	void read(const FileNode& fn)
	{
		fn["vocabSize"] >> vocabSize;
		fn["memoryUse"] >> memoryUse;
		fn["descProportion"] >> descProportion;
	}
	void write(FileStorage& fs) const
	{
		fs << "vocabSize" << vocabSize;
		fs << "memoryUse" << memoryUse;
		fs << "descProportion" << descProportion;
	}
	void print() const
	{
		cout << "vocabSize: " << vocabSize << endl;
		cout << "memoryUse: " << memoryUse << endl;
		cout << "descProportion: " << descProportion << endl;
	}

	// It shouldn't matter which object class is specified here - visual vocab will still be the same.
	int vocabSize; //number of visual words in vocabulary to train
	int memoryUse; // Memory to preallocate (in MB) when training vocab.
	// Change this depending on the size of the dataset/available memory.
	float descProportion; // Specifies the number of descriptors to use from each image as a proportion of the total num descs.
};

struct SVMTrainParamsExt
{
	SVMTrainParamsExt() : descPercent(0.5f), targetRatio(0.4f), balanceClasses(true) {}
	SVMTrainParamsExt(float _descPercent, float _targetRatio, bool _balanceClasses) :
		descPercent(_descPercent), targetRatio(_targetRatio), balanceClasses(_balanceClasses) {}
	void read(const FileNode& fn)
	{
		fn["descPercent"] >> descPercent;
		fn["targetRatio"] >> targetRatio;
		fn["balanceClasses"] >> balanceClasses;
	}
	void write(FileStorage& fs) const
	{
		fs << "descPercent" << descPercent;
		fs << "targetRatio" << targetRatio;
		fs << "balanceClasses" << balanceClasses;
	}
	void print() const
	{
		cout << "descPercent: " << descPercent << endl;
		cout << "targetRatio: " << targetRatio << endl;
		cout << "balanceClasses: " << balanceClasses << endl;
	}

	float descPercent; // Percentage of extracted descriptors to use for training.
	float targetRatio; // Try to get this ratio of positive to negative samples (minimum).
	bool balanceClasses;    // Balance class weights by number of samples in each (if true cSvmTrainTargetRatio is ignored).
};


BagOfWords::BagOfWords() 
{
}

BagOfWords::~BagOfWords() 
{
}



static void readUsedParams(const FileNode& fn, string& vocName, DDMParams& ddmParams, VocabTrainParams& vocabTrainParams, SVMTrainParamsExt& svmTrainParamsExt)
{
	fn["vocName"] >> vocName;

	FileNode currFn = fn;

	currFn = fn["ddmParams"];
	ddmParams.read(currFn);

	currFn = fn["vocabTrainParams"];
	vocabTrainParams.read(currFn);

	currFn = fn["svmTrainParamsExt"];
	svmTrainParamsExt.read(currFn);
}


static Ptr<Feature2D> createByName(const String& name)
{
	if (name == "SIFT")
		return SIFT::create();
	if (name == "SURF")
		return SURF::create();
	if (name == "ORB")
		return ORB::create();
	if (name == "BRISK")
		return BRISK::create();
	if (name == "KAZE")
		return KAZE::create();
	if (name == "AKAZE")
		return AKAZE::create();
	return Ptr<Feature2D>();
}


static void makeUsedDirs(const string& rootPath, DDMParams ddmParams)
{
	string tmpPath = rootPath + ddmParams.detectorType + "/";
	utils::mkdir(tmpPath + bowImageDescriptorsDir);
	utils::mkdir(tmpPath + svmsDir);
	//utils::mkdir(tmpPath + plotsDir);
}

static bool readVocabulary(const string& filename, Mat& vocabulary)
{
	cout << "Reading vocabulary...";
	FileStorage fs(filename, FileStorage::READ);
	if (fs.isOpened())
	{
		fs["vocabulary"] >> vocabulary;
		cout << "done" << endl;
		return true;
	}
	return false;
}

static bool writeVocabulary(const string& filename, const Mat& vocabulary)
{
	cout << "Saving vocabulary..." << endl;
	FileStorage fs(filename, FileStorage::WRITE);
	if (fs.isOpened())
	{
		fs << "vocabulary" << vocabulary;
		return true;
	}
	return false;
}

static Mat trainVocabulary(const string& filename, vector<string>& images, const VocabTrainParams& trainParams,
	const Ptr<FeatureDetector>& fdetector, const Ptr<DescriptorExtractor>& dextractor)
{
	Mat vocabulary;
	if (!readVocabulary(filename, vocabulary))
	{
		CV_Assert(dextractor->descriptorType() == CV_32FC1);
		const int elemSize = CV_ELEM_SIZE(dextractor->descriptorType());
		const int descByteSize = dextractor->descriptorSize() * elemSize;
		const int bytesInMB = 1048576;
		const int maxDescCount = (trainParams.memoryUse * bytesInMB) / descByteSize; // Total number of descs to use for training.

		cout << "Computing descriptors..." << endl;
		RNG& rng = theRNG();
		TermCriteria terminate_criterion;
		terminate_criterion.epsilon = FLT_EPSILON;
		BOWKMeansTrainer bowTrainer(trainParams.vocabSize, terminate_criterion, 3, KMEANS_PP_CENTERS);

		while (images.size() > 0)
		{
			if (bowTrainer.descriptorsCount() > maxDescCount)
			{
#ifdef DEBUG_DESC_PROGRESS
				cout << "Breaking due to full memory ( descriptors count = " << bowTrainer.descriptorsCount()
					<< "; descriptor size in bytes = " << descByteSize << "; all used memory = "
					<< bowTrainer.descriptorsCount()*descByteSize << endl;
#endif
				break;
			}

			// Randomly pick an image from the dataset which hasn't yet been seen
			// and compute the descriptors from that image.
			int randImgIdx = rng((unsigned)images.size());
			Mat colorImage = imread(images[randImgIdx]);
			vector<KeyPoint> imageKeypoints;
			fdetector->detect(colorImage, imageKeypoints);
			Mat imageDescriptors;
			dextractor->compute(colorImage, imageKeypoints, imageDescriptors);

			//check that there were descriptors calculated for the current image
			if (!imageDescriptors.empty())
			{
				int descCount = imageDescriptors.rows;
				// Extract trainParams.descProportion descriptors from the image, breaking if the 'allDescriptors' matrix becomes full
				int descsToExtract = static_cast<int>(trainParams.descProportion * static_cast<float>(descCount));
				// Fill mask of used descriptors
				vector<char> usedMask(descCount, false);
				fill(usedMask.begin(), usedMask.begin() + descsToExtract, true);
				for (int i = 0; i < descCount; i++)
				{
					int i1 = rng(descCount), i2 = rng(descCount);
					char tmp = usedMask[i1]; usedMask[i1] = usedMask[i2]; usedMask[i2] = tmp;
				}

				for (int i = 0; i < descCount; i++)
				{
					if (usedMask[i] && bowTrainer.descriptorsCount() < maxDescCount)
						bowTrainer.add(imageDescriptors.row(i));
				}
			}

#ifdef DEBUG_DESC_PROGRESS
			cout <<cvRound((static_cast<double>(bowTrainer.descriptorsCount()) / static_cast<double>(maxDescCount))*100.0)
				<< " % memory used" << (imageDescriptors.empty() ? " -> no descriptors extracted, skipping" : "") << endl;
#endif

			// Delete the current element from images so it is not added again
			images.erase(images.begin() + randImgIdx);
		}

		cout << "Maximum allowed descriptor count: " << maxDescCount << ", Actual descriptor count: " << bowTrainer.descriptorsCount() << endl;

		cout << "Training vocabulary..." << endl;
		vocabulary = bowTrainer.cluster();

		if (!writeVocabulary(filename, vocabulary))
		{
			cout << "Error: file " << filename << " can not be opened to write" << endl;
			exit(-1);
		}
	}
	return vocabulary;
}


// Load in the bag of words vectors for a set of images, from file if possible
static void calculateImageDescriptors(const vector<ObdImage>& images, vector<Mat>& imageDescriptors,
	Ptr<BOWImgDescriptorExtractor>& bowExtractor, const Ptr<FeatureDetector>& fdetector,const string& resPath)
{
	CV_Assert(!bowExtractor->getVocabulary().empty());
	imageDescriptors.resize(images.size());

	for (size_t i = 0; i < images.size(); i++)
	{
		string filename = resPath + bowImageDescriptorsDir + "/" + images[i].id + ".xml.gz";

		FileStorage fs(filename, FileStorage::READ);

		if (fs.isOpened()) 
		{
			fs["imageDescriptor"] >> imageDescriptors[i];
#ifdef DEBUG_DESC_PROGRESS
			cout << "Loaded bag of word vector for image " << i + 1 << " of " << images.size() << " (" << images[i].id << ")" << endl;
#endif
		}
		else
		{
			Mat colorImage = imread(images[i].path);
#ifdef DEBUG_DESC_PROGRESS
			cout << "Computing descriptors for image " << i + 1 << " of " << images.size() << " (" << images[i].id << ")" << flush;
#endif
			vector<KeyPoint> keypoints;
			fdetector->detect(colorImage, keypoints);
#ifdef DEBUG_DESC_PROGRESS
			cout << " + generating BoW vector" << std::flush;
#endif
			bowExtractor->compute(colorImage, keypoints, imageDescriptors[i]);
#ifdef DEBUG_DESC_PROGRESS
			cout << " ...DONE " << static_cast<int>(static_cast<float>(i + 1) / static_cast<float>(images.size())*100.0)
				<< " % complete" << endl;
#endif
			if (!imageDescriptors[i].empty())
			{
				FileStorage fs(filename, FileStorage::WRITE);
				if (fs.isOpened())
				{
					fs << "imageDescriptor" << imageDescriptors[i];
				}
				else 
				{
					cout << "Error: file " << filename << "can not be opened to write bow image descriptor" << endl;
					//exit(-1);
				}
			}
		}
	}
}

	
static void removeEmptyBowImageDescriptors(vector<ObdImage>& images, vector<Mat>& bowImageDescriptors,
	vector<char>& objectPresent)
{
	CV_Assert(!images.empty());
	for (int i = (int)images.size() - 1; i >= 0; i--)
	{
		bool res = bowImageDescriptors[i].empty();
		if (res)
		{
			cout << "Removing image " << images[i].id << " due to no descriptors..." << endl;
			images.erase(images.begin() + i);
			bowImageDescriptors.erase(bowImageDescriptors.begin() + i);
			objectPresent.erase(objectPresent.begin() + i);
		}
	}
}

static void removeBowImageDescriptorsByCount(vector<ObdImage>& images, vector<Mat> bowImageDescriptors, vector<char> objectPresent,
	const SVMTrainParamsExt& svmParamsExt, int descsToDelete)
{
	RNG& rng = theRNG();
	int pos_ex = (int)std::count(objectPresent.begin(), objectPresent.end(), (char)1);
	int neg_ex = (int)std::count(objectPresent.begin(), objectPresent.end(), (char)0);

	while (descsToDelete != 0)
	{
		int randIdx = rng((unsigned)images.size());

		// Prefer positive training examples according to svmParamsExt.targetRatio if required
		if (objectPresent[randIdx])
		{
			if ((static_cast<float>(pos_ex) / static_cast<float>(neg_ex + pos_ex)  < svmParamsExt.targetRatio) &&
				(neg_ex > 0) && (svmParamsExt.balanceClasses == false))
			{
				continue;
			}
			else
			{
				pos_ex--;
			}
		}
		else
		{
			neg_ex--;
		}

		images.erase(images.begin() + randIdx);
		bowImageDescriptors.erase(bowImageDescriptors.begin() + randIdx);
		objectPresent.erase(objectPresent.begin() + randIdx);

		descsToDelete--;
	}
	CV_Assert(bowImageDescriptors.size() == objectPresent.size());
}

static void setSVMParams(Ptr<SVM> & svm, const Mat& responses, bool balanceClasses)
{
	int pos_ex = countNonZero(responses == 1);
	int neg_ex = countNonZero(responses == -1);
	cout << pos_ex << " positive training samples; " << neg_ex << " negative training samples" << endl;

	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::RBF);
	if (balanceClasses)
	{
		Mat class_wts(2, 1, CV_32FC1);
		// The first training sample determines the '+1' class internally, even if it is negative,
		// so store whether this is the case so that the class weights can be reversed accordingly.
		bool reversed_classes = (responses.at<float>(0) < 0.f);
		if (reversed_classes == false)
		{
			class_wts.at<float>(0) = static_cast<float>(pos_ex) / static_cast<float>(pos_ex + neg_ex); // weighting for costs of positive class + 1 (i.e. cost of false positive - larger gives greater cost)
			class_wts.at<float>(1) = static_cast<float>(neg_ex) / static_cast<float>(pos_ex + neg_ex); // weighting for costs of negative class - 1 (i.e. cost of false negative)
		}
		else
		{
			class_wts.at<float>(0) = static_cast<float>(neg_ex) / static_cast<float>(pos_ex + neg_ex);
			class_wts.at<float>(1) = static_cast<float>(pos_ex) / static_cast<float>(pos_ex + neg_ex);
		}
		svm->setClassWeights(class_wts);
	}
}

static void setSVMTrainAutoParams(ParamGrid& c_grid, ParamGrid& gamma_grid,
	ParamGrid& p_grid, ParamGrid& nu_grid,
	ParamGrid& coef_grid, ParamGrid& degree_grid)
{
	c_grid = SVM::getDefaultGrid(SVM::C);

	gamma_grid = SVM::getDefaultGrid(SVM::GAMMA);

	p_grid = SVM::getDefaultGrid(SVM::P);
	p_grid.logStep = 0;

	nu_grid = SVM::getDefaultGrid(SVM::NU);
	nu_grid.logStep = 0;

	coef_grid = SVM::getDefaultGrid(SVM::COEF);
	coef_grid.logStep = 0;

	degree_grid = SVM::getDefaultGrid(SVM::DEGREE);
	degree_grid.logStep = 0;
}

static Ptr<SVM> trainSVMClassifier(const SVMTrainParamsExt& svmParamsExt, const DDMParams& ddmParams, vector<ObdImage> images,
	vector<char> objectPresent,	Ptr<BOWImgDescriptorExtractor>& bowExtractor, const Ptr<FeatureDetector>& fdetector,const string& resPath)
{
	/* first check if a previously trained svm for the current class has been saved to file */
	string svmFilename = resPath + svmsDir + "/" + ddmParams.descriptorType + ".xml.gz";
	Ptr<SVM> svm;

	FileStorage fs(svmFilename, FileStorage::READ);
	if (fs.isOpened())
	{
		cout << "*** LOADING SVM CLASSIFIER FOR" << ddmParams.descriptorType << " ***" << endl;
		svm = StatModel::load<SVM>(svmFilename);
	}
	else
	{
		cout << "*** TRAINING CLASSIFIER FOR  " << ddmParams.descriptorType << " ***" << endl;
		cout << "CALCULATING BOW VECTORS FOR TRAINING SET OF " << ddmParams.descriptorType << "..." << endl;

		// Get classification ground truth for images in the training set
		//vector<ObdImage> images;
		vector<Mat> bowImageDescriptors;
		//vector<char> objectPresent = objectPresent;
		//vocData.getClassImages(objClassName, CV_OBD_TRAIN, images, objectPresent);

		// Compute the bag of words vector for each image in the training set.
		calculateImageDescriptors(images, bowImageDescriptors, bowExtractor, fdetector, resPath);

		// Remove any images for which descriptors could not be calculated
		removeEmptyBowImageDescriptors(images, bowImageDescriptors, objectPresent);

		CV_Assert(svmParamsExt.descPercent > 0.f && svmParamsExt.descPercent <= 1.f);
		if (svmParamsExt.descPercent < 1.f)
		{
			int descsToDelete = static_cast<int>(static_cast<float>(images.size())*(1.0 - svmParamsExt.descPercent));

			cout << "Using " << (images.size() - descsToDelete) << " of " << images.size() <<
				" descriptors for training (" << svmParamsExt.descPercent*100.0 << " %)" << endl;
			removeBowImageDescriptorsByCount(images, bowImageDescriptors, objectPresent, svmParamsExt, descsToDelete);
		}

		// Prepare the input matrices for SVM training.
		Mat trainData((int)images.size(), bowExtractor->getVocabulary().rows, CV_32FC1);
		Mat responses((int)images.size(), 1, CV_32SC1);

		// Transfer bag of words vectors and responses across to the training data matrices
		for (size_t imageIdx = 0; imageIdx < images.size(); imageIdx++)
		{
			// Transfer image descriptor (bag of words vector) to training data matrix
			Mat submat = trainData.row((int)imageIdx);
			if (bowImageDescriptors[imageIdx].cols != bowExtractor->descriptorSize())
			{
				cout << "Error: computed bow image descriptor size " << bowImageDescriptors[imageIdx].cols
					<< " differs from vocabulary size" << bowExtractor->getVocabulary().cols << endl;
			}
			bowImageDescriptors[imageIdx].copyTo(submat);

			// Set response value
			responses.at<int>((int)imageIdx) = objectPresent[imageIdx] ? 1 : -1;
		}

		cout << "TRAINING SVM FOR CLASS ..." << ddmParams.descriptorType << "..." << endl;
		svm = SVM::create();
		setSVMParams(svm, responses, svmParamsExt.balanceClasses);
		ParamGrid c_grid, gamma_grid, p_grid, nu_grid, coef_grid, degree_grid;
		setSVMTrainAutoParams(c_grid, gamma_grid, p_grid, nu_grid, coef_grid, degree_grid);

		svm->trainAuto(TrainData::create(trainData, ROW_SAMPLE, responses), 10,
			c_grid, gamma_grid, p_grid, nu_grid, coef_grid, degree_grid);
		cout << "SVM TRAINING FOR CLASS " << ddmParams.descriptorType << " COMPLETED" << endl;

		svm->save(svmFilename);
		cout << "SAVED CLASSIFIER TO FILE" << endl;
	}
	return svm;
}

static void convertImageCodesToObdImages(const vector<string>& imagesPath, vector<ObdImage>& images)
{
	images.clear();
	images.reserve(imagesPath.size());

	string id;
	//transfer to output arrays
	for (size_t i = 0; i < imagesPath.size(); ++i)
	{
		//generate image path and indices from extracted string code
		id = Utils::getFileName(imagesPath[i]);
		images.push_back(ObdImage(id, imagesPath[i]));
	}
}

static void computeConfidences(const Ptr<SVM>& svm, vector<string>& imageTest, vector<char> objectPresent,
	Ptr<BOWImgDescriptorExtractor>& bowExtractor, const Ptr<FeatureDetector>& fdetector,
	const string& resPath)
{
	// Get classification ground truth for images in the test set
	vector<ObdImage> images;
	vector<Mat> bowImageDescriptors;

	convertImageCodesToObdImages(imageTest, images);
	//vector<char> objectPresent;
	//vocData.getClassImages(objClassName, CV_OBD_TEST, images, objectPresent);

	// Compute the bag of words vector for each image in the test set
	calculateImageDescriptors(images, bowImageDescriptors, bowExtractor, fdetector, resPath);
	// Remove any images for which descriptors could not be calculated
	removeEmptyBowImageDescriptors(images, bowImageDescriptors, objectPresent);

	double count_all = images.size();
	double ptrue_rtrue = 0;
	double ptrue_rfalse = 0;
	double pfalse_rtrue = 0;
	double pfalse_rfalse = 0;

	for (size_t imageIdx = 0; imageIdx < count_all; imageIdx++)
	{
		auto predict = int(svm->predict(bowImageDescriptors[imageIdx]));

		if (predict == -1)
		{
			predict = 0;
		}
		std::cout << "predict: " << predict << std::endl;

		auto real = objectPresent[imageIdx];
		if (predict == 1 && real == 1) ptrue_rtrue++;
		if (predict == 1 && real == 0) ptrue_rfalse++;
		if (predict == 0 && real == 1) pfalse_rtrue++;
		if (predict == 0 && real == 0) pfalse_rfalse++;
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
	}
	else {
		std::cout << "recall: "
			<< "NA" << std::endl;
	}

	double Fscore = 0;
	if (precise + recall != 0) {
		Fscore = 2 * (precise * recall) / (precise + recall);
		std::cout << "Fscore: " << Fscore << std::endl;
	}
	else {
		std::cout << "Fscore: "
			<< "NA" << std::endl;
	}
}


void BagOfWords::getVocabulary(vector<string>& imagePath, vector<char> objectPresent, vector<string>& imageTest, vector<char> objPresentTest)
{
	//const string resPath = "D:/freelancer/201611/SVMExample";

	string vocName;
	DDMParams ddmParams;
	VocabTrainParams vocabTrainParams;
	SVMTrainParamsExt svmTrainParamsExt;
	vector<string> vabImgPath = imagePath;
	vector<ObdImage> images;

	

	FileStorage paramsFS(paramsFile, FileStorage::READ);
	if (paramsFS.isOpened())
	{
		readUsedParams(paramsFS.root(), vocName, ddmParams, vocabTrainParams, svmTrainParamsExt);
		makeUsedDirs(resPath, ddmParams);
	}
	else
	{
		cout << "Error: file " << paramsFile << " can not be opened or don't exist" << endl;
	}

	// Create detector, descriptor, matcher.
	if (ddmParams.detectorType != ddmParams.descriptorType)
	{
		cout << "detector and descriptor should be the same\n";
	}
	Ptr<Feature2D> featureDetector = createByName(ddmParams.detectorType);
	Ptr<DescriptorExtractor> descExtractor = featureDetector;
	Ptr<BOWImgDescriptorExtractor> bowExtractor;
	if (!featureDetector || !descExtractor)
	{
		cout << "featureDetector or descExtractor was not created" << endl;
	}
	{
		Ptr<DescriptorMatcher> descMatcher = DescriptorMatcher::create(ddmParams.matcherType);
		if (!featureDetector || !descExtractor || !descMatcher)
		{
			cout << "descMatcher was not created" << endl;
		}
		bowExtractor = makePtr<BOWImgDescriptorExtractor>(descExtractor, descMatcher);
	}
	
	convertImageCodesToObdImages(imagePath, images);

	// 1. Train visual word vocabulary if a pre-calculated vocabulary file doesn't already exist from previous run
	
	Mat vocabulary = trainVocabulary(resPath + "/" + vocabularyFile, vabImgPath, vocabTrainParams,
	featureDetector, descExtractor);
	bowExtractor->setVocabulary(vocabulary);

	// Train a classifier on train dataset
	Ptr<SVM> svm = trainSVMClassifier(svmTrainParamsExt, ddmParams, images, objectPresent,
		bowExtractor, featureDetector, resPath);

	computeConfidences(svm, imageTest, objPresentTest, bowExtractor, featureDetector, resPath);
	
}