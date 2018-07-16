#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
__global__ void AdamWRUpdate(int N, Dtype* theta, Dtype* g, Dtype* m, Dtype* v,
    Dtype beta1, Dtype beta2, Dtype eps_hat, Dtype corrected_local_rate, Dtype local_decay, Dtype yita) {
  CUDA_KERNEL_LOOP(i, N) {
    float gi = g[i];
    float mi = m[i] = m[i]*beta1 + gi*(1-beta1);
    float vi = v[i] = v[i]*beta2 + gi*gi*(1-beta2);
    g[i] = yita * ((corrected_local_rate * mi / (sqrt(vi) + eps_hat)) + local_decay * theta[i]);
    // g[i] = corrected_local_rate * mi / (sqrt(vi) + eps_hat);
  }
}
template <typename Dtype>
void adamwr_update_gpu(int N, Dtype *theta, Dtype* g, Dtype* m, Dtype* v, Dtype beta1,
    Dtype beta2, Dtype eps_hat, Dtype corrected_local_rate, Dtype local_decay, Dtype yita) {

  AdamWRUpdate<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, theta, g, m, v, beta1, beta2, eps_hat, corrected_local_rate, local_decay, yita);
  CUDA_POST_KERNEL_CHECK;
}

template void adamwr_update_gpu<float>(int, float*, float*, float*, float*,
    float, float, float, float, float, float);
template void adamwr_update_gpu<double>(int, double*, double*, double*, double*,
    double, double, double, double, double, double);

}  // namespace caffe