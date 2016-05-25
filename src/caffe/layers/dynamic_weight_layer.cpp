#include <vector>

#include "caffe/layers/dynamic_weight_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void DynamicWeightLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void DynamicWeightLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  vector<int> top_shape =  bottom[0]->shape() ;
  top[0]->Reshape(top_shape);

  top[0]->ShareData(*bottom[0]);
  top[0]->ShareDiff(*bottom[0]);

}

template <typename Dtype>
void DynamicWeightLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	int weightIndex = bottom.size() - 1 ;
	int numPipeline = bottom.size() - 1 ;
	int num =  bottom[0]->shape(0);
	int channels = bottom[0]->shape(1) ;

	Dtype* top_data = top[0]->mutable_cpu_data();
	Dtype weight_sum =  0 ;
	const Dtype* weight_data =  bottom[weightIndex]->cpu_data() ;



	for( int i = 0 ; i<num ; ++i ){
		for( int j = 0 ; j < channels ; j++ ){
			top_data[i*channels + j] = 0 ;

			weight_sum = 0 ;
			for(int k = 0 ; k < numPipeline ; k++){
				weight_sum+= weight_data[i*channels + j*numPipeline + k];
			}
			for(int k = 0 ; k < numPipeline ; k++){
				const Dtype* bottom_data =  bottom[k]->cpu_data() ;
				top_data[i*channels + j] +=   bottom_data[i*channels + j] * (weight_data[i*channels + j*numPipeline + k]/weight_sum);

			}

			LOG(INFO) << "Weight Values "<< (j+1) << " " << (double)weight_data[i*channels + j*numPipeline + 0]<<" "<< (double)weight_data[i*channels + j*numPipeline + 1];
		}
	}

	LOG(INFO)<< "Forward Done" ;




}

template <typename Dtype>
void DynamicWeightLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	int weightIndex = bottom.size() - 1 ;
	int numPipeline = bottom.size() - 1 ;

	int num =  top[0]->shape(0);
	int channels = top[0]->shape(1) ;

	Dtype weight_sum =  0 ;


	const Dtype* top_diff = top[0]->cpu_diff();
	const Dtype* weight_data =  bottom[weightIndex]->cpu_data() ;


	/*
	 * Setting differenciation for the pipelines
	 */

	for( int i = 0 ; i<num ; i++ ){
		for( int j = 0 ; j < channels ; j++ ){

			weight_sum = 0 ;
			for(int k = 0 ; k < numPipeline ; k++){
				weight_sum+= weight_data[i*channels + j*numPipeline + k];
			}

			for(int k = 0 ; k < numPipeline ; k++){


				Dtype* bottom_diff =  bottom[k]->mutable_cpu_diff();
				bottom_diff[i*channels + j] =   top_diff[i*channels + j]*(weight_data[i*channels + j*numPipeline + k]/weight_sum);


			}
			/*
			LOG(INFO) << "Weight Values " << (double)weight_data[i*numPipeline + 0]/weight_sum<<" "<< (double)weight_data[i*numPipeline + 1]/weight_sum<<" "\
			<<bottom.size();
			*/
		}
	}

	/*
	* Setting differenciation for the weight pipeline
	*/
	Dtype* weight_diff =  bottom[weightIndex]->mutable_cpu_diff() ;
	for( int i = 0 ; i<num ; i++ ){
			for( int j = 0 ; j < channels ; j++ ){

				weight_sum = 0 ;

				for(int k = 0 ; k < numPipeline ; k++){
					weight_sum+= weight_data[i*channels + j*numPipeline + k];
				}

				for(int k = 0 ; k < numPipeline ; k++){

					const Dtype* bottom_data =  bottom[k]->cpu_data() ;
					weight_diff[i*channels + j*numPipeline + k] =   top_diff[i*channels + j]*bottom_data[i*channels + j]*((weight_sum - weight_data[i*channels + j*numPipeline + k])/(weight_sum*weight_sum)) ;

				}

			}
		}



	/*

	Dtype* weight_diff =  bottom[weightIndex]->mutable_cpu_diff() ;
	for( int i = 0 ; i<num ; i++ ){
		weight_sum = 0 ;
		for(int j = 0 ; j < numPipeline ; j++){
			weight_sum+= weight_data[i*numPipeline + j];
		}

		for( int j = 0 ; j < numPipeline; j++ ){
			weight_diff[i*numPipeline + j] = 0 ;
			const Dtype* bottom_data =  bottom[j]->cpu_data() ;
			for(int k = 0 ; k < channels ; k++){
				weight_diff[i*numPipeline + j] +=  top_diff[i*channels + k]*bottom_data[i*channels + k] * (weight_sum - weight_data[i*numPipeline + j])/(weight_sum*weight_sum) ;
			}
		}
	}
    */



}

#ifdef CPU_ONLY
STUB_GPU(DynamicWeightLayer);
#endif

INSTANTIATE_CLASS(DynamicWeightLayer);
REGISTER_LAYER_CLASS(DynamicWeight);

}  // namespace caffe
