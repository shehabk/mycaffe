/**
 *The adaboost algorithm file
 *@author Shizhong Han
 *@verion 1.0 03/11/2014
 */
#ifndef ADABOOSTBINARY_H
#define ADABOOSTBINARY_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include "cxcore.h"
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <list>
#include <numeric>
#include <omp.h>
#include <algorithm>
#include "caffe/util/basefunc.hpp"
#define SIZE 200
using namespace std;
using namespace cv;
/**
 *Binary tree weak classifier for AdaBoost
 */
namespace caffe {

/**
 *Binary tree weak classifier for AdaBoost
 */
typedef struct tag_binaryTreeClassifier
{
    int x;//x position in face image
    int y;//y position in face image
    int bindex;//The index for current selected weak classifier
    double alpha;//The alpha value
    double pro_thresh;//The threshold for the bianry tree weak classifier
    int thr_sign;//The sign for the threshold
    double minerror;//The error in current iteration
    double real_margin;
}lda_classifier;

class adaBoostBinary
{
private:
	CvMat *weight;
	float *iter_res;
    int *labelsign;
	int maxIterationNum;
	int pos_sample_num;
	int neg_sample_num;
	double rate_margin;


public:
	 vector<lda_classifier> strongclassifier;
	 adaBoostBinary();
     virtual ~adaBoostBinary();
    /**
     *call the memory for the adaboost variable
     *@param posnum The positive sample number
     *@param negnum The nigative sample number
     *@param iterNum The iteration number
     *@return 1 If it is sucessfull
     */

    int allocateMemory(int posnum, int negnum, int iterNum);
     /**
     *Initialize weight matrix using the sumation value of sum 
     *@param weight_mat The weight matrix for storing the weight value
     *@param sample_num The sample number for intialization.
     *@param sum The sumation value for all the weight after initialization
     *@return 1 If it is sucessfull
     */
    int Initialize(CvMat *label);
    int InitializeWeightFile(CvMat *label);
    /**
     *Choose the weak classifier using the fast method just based on the distribution of the data
     *@param data_pos The positive data matrix
     *@param data_neg The negative data matrix
     *@param weight_pos The positive weight matrix
     *@param weight_neg The negative weight matrix
     *@param current_best_classifier The current weak classifier for selection.
     *@param best_res_pos The classification result for current weak classifier
     *@param best_res_neg The classification result for current weak classifier.
     *@param bindexlabel The valid bindex label for selection
     */
    int choosefeature_fast(CvMat *data,CvMat *label, lda_classifier *current_best_classifier, bool *bindexlabel, float imith, int curIteration);

    template <typename Dtype>
    int choosefeature_fastupdate(CvMat *data,CvMat *label, lda_classifier *current_best_classifier, bool* bindexlabel,Dtype *sign,Dtype *thresh, float imith, int curIteration);
    /**
     *Load the feature continually
     *@param feature_positive The positive feature
     *@param feature_negative The negative feature
     *@param trainfile The file path for reading the train feature
     *@param labelmat The label mat
     *@param auindex The current auindex
     */
    int LoadDataAllFeat_set(CvMat *feature_train, const char* trainfile,int ib, int id,int dim,int &inum);
    
    /**
     *update the weight matrix
     *@param feature_positive The positive feature data
     *@param feature_negative The negative feature data
     *@param current_best_classifier The current best weak classifier
     */
    int updataWeight(CvMat *feature_train,lda_classifier *current_best_classifier, float imith, int curIteration);
    /**
     *Test the classification performance in the train data
     */
    float testStrongClassifier();



};


}

#endif


