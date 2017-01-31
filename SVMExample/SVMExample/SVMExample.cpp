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

using namespace cv;
string resPath;

int main(int argc, char** argv)
{	
    SvmTrain svmTr = SvmTrain();
	
    FileStorage paramsFS("params.xml", FileStorage::READ);
	
	if (paramsFS.isOpened())
	{
		FileNode currFn = paramsFS["resPath"];
		
		currFn >> resPath;
		
		currFn = paramsFS["featureType"];
		string featureType = "";
		currFn >> featureType;
		
		if((featureType == "LBP") || (featureType == "HOG"))
		{
			svmTr.feaType = featureType;
			
			svmTr.trainAuto();
		}
		else
		{
			svmTr.getBOWFeatures();
		}
	}
	else
	{
		//params.xml should be in the same path with SVMExample.vcxproj
		cout << "Open params.xml fail, please make sure the params.xml in the right path!" << endl;
		
	}
	
	//keep console 
    cin.get();
	
	return 0;
}

