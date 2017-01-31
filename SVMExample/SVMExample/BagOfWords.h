#ifndef __BOWOFWORDS_H__
#define __BOWOFWORDS_H__

#include "stdafx.h"
#include <vector>
using namespace std;

class BagOfWords
{
	
public:

	BagOfWords();

	~BagOfWords();

	void getVocabulary(vector<string>& image_codes, vector<char> objectPresent, vector<string>& imageTest, vector<char> objPresentTest);

};

#endif

