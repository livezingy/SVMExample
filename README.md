# SVMExample
1. This repository test on VS2013_X64 + opencv3.1 + opencv_contrib. The opencv3.1 and opencv_contrib compiles from source code.

2. In EasyPR（an open source license plate recognition system）, SVM (support vector machine) is used to select the real license plate in the candidate license plate. The LBP feature is as input of SVM in EasyPR1.4. This repository comes from the svm_train.cpp in EasyPR1.4, But it added HOG/SIFT/SURF feature as input of SVM. 

3. The routine of running repository: set parameters in param.xml in the same directory as the project file .vcproj in the current project.  There is a param.xml in the the folder you could refer to.

4. More information about SVMExample refer to http://livezingy.com/sift-surf-bag-of-words-example/

![The interface of CaptchaProcess](http://livezingy.qiniudn.com/201701/opencv/featureCompare.png)
![The flow chart of SVMExample](http://livezingy.qiniudn.com/201701/opencv/SvmExample.PNG)
