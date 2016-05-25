#ifndef CAFFE_BOOSTING_LAYER_HPP_
#define CAFFE_BOOSTING_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Also known as a "fully-connected" layer, computes an inner product
 *        with a set of learned weights, and (optionally) adds biases.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
/**
 * @brief Also known as a "boosting layer" layer, computes an boosting classifer
 *        with a set of learned weights, and (optionally) adds biases.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class BoostingLayer : public Layer<Dtype> {
 public:
  explicit BoostingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Boosting"; }

  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 3; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void adaboost(const Dtype* bottom_data, const Dtype* bottom_label, Dtype* alpha, Dtype* thresh, Dtype* sign);
 // virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
 //     const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  //virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
  //    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int M_;
  int K_;
  int N_;
  float imith_;
  float imithrate_;
  string bottomdatafile_;
  string weakclassifierfile_;
  string bottomdifffile_;
  string strongscorefile_;
  string weakscorefile_;
  string strongdifffile_;
  string weakdifffile_;
  string strongratefile_;
  int ite_;
};


}  // namespace caffe

#endif  // CAFFE_BOOSTING_LAYER_HPP_
