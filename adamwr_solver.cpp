//
// Created by linxiaodong on 18-7-12.
//

#include <vector>
#include <math.h>
#include "caffe/sgd_solvers.hpp"

namespace caffe {

    template<typename Dtype>
    void AdamWRSolver<Dtype>::AdamWRPreSolve() {
        // Add the extra history entries for Adam after those from
        // SGDSolver::PreSolve
        const vector<Blob<Dtype> *> &net_params = this->net_->learnable_params();
        for (int i = 0; i < net_params.size(); ++i) {
            const vector<int> &shape = net_params[i]->shape();
            this->history_.push_back(
                    shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
        }
    }


#ifndef CPU_ONLY

    template<typename Dtype>
    void adamwr_update_gpu(int N, Dtype *theta, Dtype *g, Dtype *m, Dtype *v, Dtype beta1,
                           Dtype beta2, Dtype eps_hat, Dtype corrected_local_rate, Dtype local_decay, Dtype yita);

#endif

    template<typename Dtype>
    void AdamWRSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate) {
        const vector<Blob<Dtype> *> &net_params = this->net_->learnable_params();
        const vector<float> &net_params_lr = this->net_->params_lr();
        Dtype local_rate = rate * net_params_lr[param_id];
        const Dtype beta1 = this->param_.momentum();
        const Dtype beta2 = this->param_.momentum2();

        // we create aliases for convenience
        size_t update_history_offset = net_params.size();
        Blob<Dtype> *val_m = this->history_[param_id].get();
        Blob<Dtype> *val_v = this->history_[param_id + update_history_offset].get();
        Blob<Dtype> *val_t = this->temp_[param_id].get();

        const int t = this->iter_ + 1;
        const Dtype correction = std::sqrt(Dtype(1) - pow(beta2, t)) /
                                 (Dtype(1.) - pow(beta1, t));
        const int N = net_params[param_id]->count();
        const Dtype eps_hat = this->param_.delta();

        Dtype yita(1.);
        bool with_restart = this->param_.with_restart();

        // false: AdamW, true: AdamWR
        if (with_restart) {
            // calculate schedule multiplier, yita.
            const Dtype cosine_decay_steps(this->param_.cosine_decay_steps());
            const Dtype cosine_decay_mult(this->param_.cosine_decay_mult());
            const int index = std::floor(
                    std::log(std::floor(this->iter_ / cosine_decay_steps) + 1) / log(cosine_decay_mult));

            Dtype T_pre(0);

            for (int i = 0; i < index; ++i) {
                T_pre += Dtype(std::pow(cosine_decay_mult, i));
            }

            T_pre *= cosine_decay_steps;

            Dtype T_i(std::pow(cosine_decay_mult, index) * cosine_decay_steps);
            Dtype T_cur = Dtype(this->iter_) - T_pre;

            CHECK_GE(T_i, T_cur) << "schedule multiplier should make sure T_i is larger than T_cur.";

            // yita = 0.5 + 0.5 * cos(pi * T_cur / T_i)
            yita = Dtype(0.5 + 0.5 * std::cos((std::acos((double(-1))) * T_cur) / T_i));
        }

        CHECK_LE(yita, Dtype(1.)) << "yita should be smaller than 1.";
        CHECK_GE(yita, Dtype(0.)) << "yita should be larger than 0.";

        // local weight decay
        const vector<float> &net_params_weight_decay =
                this->net_->params_weight_decay();
        Dtype weight_decay = this->param_.weight_decay();
        Dtype local_decay = weight_decay * net_params_weight_decay[param_id];

        switch (Caffe::mode()) {
            case Caffe::CPU: {
                // update m <- \beta_1 m_{t-1} + (1-\beta_1)g_t
                caffe_cpu_axpby(N, Dtype(1) - beta1,
                                net_params[param_id]->cpu_diff(), beta1,
                                val_m->mutable_cpu_data());

                // update v <- \beta_2 m_{t-1} + (1-\beta_2)g_t^2
                caffe_mul(N,
                          net_params[param_id]->cpu_diff(),
                          net_params[param_id]->cpu_diff(),
                          val_t->mutable_cpu_data());
                caffe_cpu_axpby(N, Dtype(1) - beta2,
                                val_t->cpu_data(), beta2,
                                val_v->mutable_cpu_data());

                // set update
                caffe_powx(N,
                           val_v->cpu_data(), Dtype(0.5),
                           val_t->mutable_cpu_data());
                caffe_add_scalar(N, eps_hat, val_t->mutable_cpu_data());
                caffe_div(N,
                          val_m->cpu_data(),
                          val_t->cpu_data(),
                          val_t->mutable_cpu_data());

                /*caffe_cpu_scale(N, local_rate * correction,
                                val_t->cpu_data(),
                                net_params[param_id]->mutable_cpu_diff());*/

                // AdamWR differs from Adam
                caffe_cpu_scale(N, local_rate * correction,
                                val_t->cpu_data(),
                                val_t->mutable_cpu_data());

                // Y = alpha * X + Y
                caffe_axpy(N, local_decay, net_params[param_id]->cpu_data(), val_t->mutable_cpu_data());

                // Y = yita * X
                caffe_cpu_scale(N, yita, val_t->cpu_data(), net_params[param_id]->mutable_cpu_diff());

                break;
            }
            case Caffe::GPU: {
#ifndef CPU_ONLY
                adamwr_update_gpu(N, net_params[param_id]->mutable_gpu_data(), net_params[param_id]->mutable_gpu_diff(),
                                  val_m->mutable_gpu_data(), val_v->mutable_gpu_data(), beta1, beta2,
                                  eps_hat, local_rate * correction, local_decay, yita);

                // update m <- \beta_1 m_{t-1} + (1-\beta_1)g_t
//                caffe_gpu_axpby(N, Dtype(1) - beta1,
//                                net_params[param_id]->gpu_diff(), beta1,
//                                val_m->mutable_gpu_data());
//
//                // update v <- \beta_2 m_{t-1} + (1-\beta_2)g_t^2
//                caffe_gpu_mul(N,
//                              net_params[param_id]->gpu_diff(),
//                              net_params[param_id]->gpu_diff(),
//                              val_t->mutable_gpu_data());
//
//                caffe_gpu_axpby(N, Dtype(1) - beta2,
//                                val_t->gpu_data(), beta2,
//                                val_v->mutable_gpu_data());
//
//                // set update
//                caffe_gpu_powx(N,
//                           val_v->gpu_data(), Dtype(0.5),
//                           val_t->mutable_gpu_data());
//
//                caffe_gpu_add_scalar(N, eps_hat, val_t->mutable_gpu_data());
//
//                caffe_gpu_div(N,
//                          val_m->gpu_data(),
//                          val_t->gpu_data(),
//                          val_t->mutable_gpu_data());
//
//                // AdamWR differs from Adam
//                caffe_gpu_scale(N, local_rate * correction,
//                                val_t->gpu_data(),
//                                val_t->mutable_gpu_data());
//
//                // Y = alpha * X + Y
//                caffe_gpu_axpy(N, local_decay, net_params[param_id]->gpu_data(), val_t->mutable_gpu_data());
//
//                // Y = yita * X
//                caffe_gpu_scale(N, yita, val_t->gpu_data(), net_params[param_id]->mutable_gpu_diff());
#else
                NO_GPU;
#endif
                break;
            }
            default:
                LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
        }
    }

    INSTANTIATE_CLASS(AdamWRSolver);

    REGISTER_SOLVER_CLASS(AdamWR);

}  // namespace caffe
