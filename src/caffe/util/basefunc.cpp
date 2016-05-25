/**
 *basefunc.cpp
 *Purpose: Some usefull basic function
 *@author Shizhong Han
 *@version 1.0 03/10/2014
 */
#include<stdio.h>
#include<stdlib.h>
#include "caffe/util/basefunc.hpp"
/**
 *Get the line number of the input document filename
 *@param filename The input file name
 *@return The line number for the input filename
 */
namespace caffe {
int GetNumline(const char *filename)
{
    FILE *fp_feature;
    fp_feature=fopen(filename,"r");
    if (NULL==fp_feature)
    {
        printf("open file %s failed \n", filename);
        return -1;
    }
    
    char line[6550000];
    int numline=0;
    while(fgets(line, sizeof(line), fp_feature)!=NULL)
        numline++;

    printf("The linenum of %s is %d\n",filename,numline);
    fclose(fp_feature);
    fp_feature = NULL;

    return numline;
}
/**
 *compare function when rank numbers from min to max
 *@param a b The input mumber a and b
 *@return bool value 1 if a>b.
 */
int mintomax(const void * a, const void * b)
{
  float tmp=((scoreindex*)a)->score - ((scoreindex*)b)->score;
  if(tmp>0) return 1;
  else if(tmp==0) return 0;
  else return -1;
}
/**
 *Compare function when ranking numbers from max to min
 *@param a b The input number a and b
 *@return bool value 1 if a<b
 */
int maxtomin(const void * a, const void * b)
{
  float tmp=((scoreindex*)b)->score - ((scoreindex*)a)->score ;
  if(tmp>0) return 1;
  else if(tmp==0) return 0;
  else return -1;
}
/**
 *write the float array of score to a file called socrefile. 
 *@param scorefile The file name of socre to write.
 *@param score The number array score
 *@sample_num The length of numbers in array score
 *@return 1 if writing sccessfully
 */
int writefile(char *scorefile, float *score, int sample_num)
{
    FILE *fw_score=fopen(scorefile, "w");
    if(fw_score == NULL)
    {
	printf("failed to create the score file %s\n", scorefile);
        return 0;
    }
    for(int i=0; i<sample_num; i++)
    {
	fprintf(fw_score, "%f\n", score[i]);
    }
    fclose(fw_score);
    return 1;
}


/**
 *read the int array of numbers from a file called socrefile. 
 *@param scorefile The file name of socre to write.
 *@param score The number array score
 *@sample_num The length of numbers in array score
 *@return 1 if writing sccessfully
 */
int readfile(char *scorefile, int *score, int sample_num)
{
    FILE *fw_score=fopen(scorefile, "r");
    if(fw_score == NULL)
    {
	printf("failed to create the score file %s\n", scorefile);
        return 0;
    }
    for(int i=0; i<sample_num; i++)
    {
	fscanf(fw_score, "%d\n", &score[i]);
    }
    fclose(fw_score);
    return 1;
}

}

