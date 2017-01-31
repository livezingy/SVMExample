# SVMExample
1. 在开源的车牌识别系统EasyPR中，用SVM（支持向量机）模型甄选出候选车牌中真正的车牌。目前EasyPR1.4的SVM模型输入的是LBP特征，本代码将EasyPR的svm_train.cpp独立出来，包含SIFT和SURF结合BOW作为SVM输入，以及LBP和HOG特征作为SVM的输入。
2. 例程要求在当前项目的工程文件.vcproj相同目录下有一个名为param.xml的文件，目前例程中有一个现成的文件。该文件中包含程序产生的文件存储路径，切换特征点类型，BOW词汇表的总量等参数均需要在此文件中设置。该文件中的内容如下图所示。
<?xml version="1.0" ?>   
<opencv_storage>  

<!--path of the generated files-->
<resPath>The path of the generated files</resPath>

<!--featureType choices: LBP/HOG/SIFT/SURF/ORB/BRISK/KAZE/AKAZE-->
<featureType>SIFT</featureType>

<ddmParams><!--parameters of bag of words-->
   <detectorType>SIFT</detectorType>
   <descriptorType>SIFT</descriptorType>
   <matcherType>BruteForce</matcherType>
 </ddmParams>

<vocabTrainParams>
   <vocabSize>1000</vocabSize><!--number of visual words in vocabulary to train-->
   <memoryUse>200</memoryUse> <!--Memory to preallocate (in MB) when training vocab.-->
   <!--pecifies the number of descriptors to use from each image as a proportion of the total num descs.-->
   <descProportion>0.3</descProportion>
</vocabTrainParams>

<svmTrainParamsExt>
   <descPercent>0.5</descPercent><!-- Percentage of extracted descriptors to use for training.-->
   <targetRatio>0.4</targetRatio><!-- Try to get this ratio of positive to negative samples (minimum)-->
   <!--Balance class weights by number of samples in each (if true cSvmTrainTargetRatio is ignored)-->
   <balanceClasses>true</balanceClasses>
 </svmTrainParamsExt>
 </opencv_storage>  
