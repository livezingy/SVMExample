#ifndef EASYPR_TRAIN_SVMTRAIN_H_
#define EASYPR_TRAIN_SVMTRAIN_H_

#include <vector>
#include <opencv2/opencv.hpp>
#include "BagOfWords.h"

//int svmTrain(bool dividePrepared, bool trainPrepared);

class SvmTrain {
 public:


  SvmTrain();

  virtual void train();

  virtual void trainAuto();

  void getBOWFeatures();

  std::string feaType;

 private:
 
 	typedef enum {
		kForward = 1, // correspond to "has plate" 
		kInverse = 0  // correspond to "no plate"
	} SvmLabel;

  typedef struct {
    std::string file;
    SvmLabel label;
  } TrainItem;
 
  BagOfWords pBOW;
  
  virtual void test(); 
  
  void getHOGFeatures(const cv::Mat& image, cv::Mat& features);

  void getLBPFeatures(const cv::Mat& image, cv::Mat& features);
 
  void prepare();

 // void convert_to_ml(const std::vector< cv::Mat > & train_samples, cv::Mat& trainData);

 // void load_images(vector< Mat > & img_lst, vector< int > & lbl_lst);

 // void get_data();

  virtual cv::Ptr<cv::ml::TrainData> tdata();

  cv::Ptr<cv::ml::SVM> svm_;

  std::vector<TrainItem> train_file_list_;
  std::vector<TrainItem> test_file_list_;

  
};


#endif  // EASYPR_TRAIN_SVMTRAIN_H_
