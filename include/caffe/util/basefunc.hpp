/**
 *basefunc.h
 *Purpose: Some usefull basic function
 *@author Shizhong Han
 *@version 1.0 03/11/2014
 */
#ifndef BASEFUNC_H
#define BASEFUNC_H
#include<stdio.h>
#include<stdlib.h>
/**
 *The struct scoreindex includes two members score and index.
 */
namespace caffe {
typedef struct tag_scoreindex
{
    float score;//actual value
    int index;//integer position
}scoreindex;
/**
 *Get the line number of the input document filename
 *@param filename The input file name
 *@return The line number for the input filename
 */
int GetNumline(const char *filename);
/**
 *compare function when rank numbers from min to max
 *@param a b The input mumber a and b
 *@return bool value 1 if a>b.
 */
int mintomax(const void * a, const void *b);
/**
 *Compare function when ranking numbers from max to min
 *@param a b The input number a and b
 *@return bool value 1 if a<b
 */
int maxtomin(const void *a, const void *b);
/**
 *write the float array of score to a file called socrefile. 
 *@param scorefile The file name of socre to write.
 *@param score The number array score
 *@sample_num The length of numbers in array score
 *@return 1 if writing sccessfully
 */
int writefile(char *scorefile, float *score, int sample_num);

}

#endif


