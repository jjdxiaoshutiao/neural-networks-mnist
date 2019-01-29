#include <iostream>
#include <cstdlib>
#include <ctime>

#include <armadillo>

#include "neural_network.h"
#include "../data_loader/data_loader.h"
#include "sigmoid.h"


using namespace std;
using namespace arma;

Network::Network(const vector<uint> &sizes) {
  num_layers_ = sizes.size();
  sizes_ = sizes;
  
  for (size_t i = 1; i != num_layers_; ++i) {
    srand(time(0));
    vec bias = randn<vec>(sizes_[i]);
    biases_.push_back(bias);
    
    srand(time(0));
    mat weight = randn<mat>(sizes_[i], sizes_[i-1]);
    weights_.push_back(weight);
  }
}

Network::Network(const string &filename) {
  fstream file(filename, ios_base::in | ios_base::binary);
  int32_t magic;
  file.read(reinterpret_cast<char*> (&magic), sizeof(magic));
  if (magic == 813) {
    uint buffer;
    file.read(reinterpret_cast<char*> (&num_layers_), sizeof(num_layers_));
    for (size_t i = 0; i != num_layers_; ++i) {
      file.read(reinterpret_cast<char*> (&buffer), sizeof(buffer));
      sizes_.push_back(buffer);
    }
    for (size_t l = 0; l != num_layers_ - 1; ++l) {
      double w_buffer[sizes_[l+1]*sizes_[l]];
      double b_buffer[sizes_[l+1]];
      file.read(reinterpret_cast<char*> (w_buffer), sizeof(double) * sizes_[l+1] * sizes_[l]);
      file.read(reinterpret_cast<char*> (b_buffer), sizeof(double) * sizes_[l+1]);
      mat weight(w_buffer, sizes_[l+1], sizes_[l]);
      vec bias(b_buffer, sizes_[l+1]);
      weights_.push_back(weight);
      biases_.push_back(bias);
    }
  }
  file.close();
}

const vec Network::feedforward(vec input) const {
  for (size_t i = 0; i != num_layers_ - 1; ++i) {
    input = sigmoid(weights_[i] * input + biases_[i]);
  }
  return input;
}

void Network::SGD(vector<pair<vec, vec>> &training_data, uint epochs, uint mini_batch_size, double eta, const vector<pair<vec, vec>> &test_data) {
  for (size_t e = 0; e != epochs; ++e) {
    srand(time(0));
    random_shuffle(training_data.begin(), training_data.end(), [](int n) { return rand() % n; });
    vector<pair<vec, vec>> mini_batch;
    for (size_t i = 0; i != training_data.size() / mini_batch_size; ++i) {
      for (size_t j = 0; j != mini_batch_size; ++j) {
        mini_batch.push_back(training_data[i*mini_batch_size+j]);
      }
      update_mini_batch(mini_batch, eta);
      mini_batch.clear();
    }
    if (test_data.size() != 0) {
      cout << "Epoch " << e + 1 << " finished, validation accuracy: " << evaluate(test_data) << "/" << test_data.size() << endl;
    }
  }
}

void Network::update_mini_batch(const vector<pair<vec, vec>> &mini_batch, double eta) {
  auto nabla_b = biases_;
  auto nabla_w = weights_;
  for (size_t i = 0; i != num_layers_ - 1; ++i) {
    nabla_b[i].zeros();
    nabla_w[i].zeros();
  }
  
  for (const pair<vec, vec> &instance : mini_batch) {
    auto delta = backpropagation(instance.first, instance.second);
    for (size_t i = 0; i != num_layers_ - 1; ++i) {
      nabla_b[i] = nabla_b[i] + delta.first[i];
      nabla_w[i] = nabla_w[i] + delta.second[i];
    }
  }

  for (size_t i = 0; i != num_layers_ - 1; ++i) {
    biases_[i] = biases_[i] - (nabla_b[i] / (mini_batch.size() / eta));
    weights_[i] = weights_[i] - (nabla_w[i] / (mini_batch.size() / eta));
  }
}

pair<vector<vec>, vector<mat>> Network::backpropagation(const vec &x, const vec &y) const {
  auto nabla_b = biases_;
  auto nabla_w = weights_;
  
  vec activition = x;
  vector<vec> activitions = { activition };
  vector<vec> zs;
  for (size_t i = 0; i != num_layers_ - 1; ++i) {
    auto z = weights_[i] * activition + biases_[i];
    zs.push_back(z);
    activition = sigmoid(z);
    activitions.push_back(activition);
  }
  
  vec error = cost_derivative(activitions.back(), y) % sigmoid_prime(zs.back());
  for (size_t l = 0; l != num_layers_ - 1; ++l) {
    size_t layer = num_layers_ - l - 2;
    nabla_b[layer] = error;
    nabla_w[layer] = error * activitions[layer].t();
    if (layer > 0) {
      error = weights_[layer].t() * error % sigmoid_prime(zs[layer-1]); 
    }
  }
  
  return make_pair(nabla_b, nabla_w);
}

uint Network::evaluate(const vector<pair<vec, vec>> &test_data) const {
  uint passed = 0;
  for (auto instance : test_data) {
    if (feedforward(instance.first).index_max() == instance.second.index_max()) {
      ++passed;
    }
  }
  return passed;
}

void Network::save(const string &filename) const {
  fstream file(filename, ios_base::out | ios_base::binary);
  int32_t magic = 813;
  file.write(reinterpret_cast<char*> (&magic), sizeof(magic));
  int32_t layers = num_layers_;
  file.write(reinterpret_cast<char*> (&layers), sizeof(layers));
  for (int32_t size : sizes_) {
    file.write(reinterpret_cast<char*> (&size), sizeof(size));
  }
  double buffer;
  for (size_t i = 0; i != num_layers_ - 1; ++i) {
    const double *weight_ptr = weights_[i].memptr();
    const double *bias_ptr = biases_[i].memptr();
    file.write(reinterpret_cast<const char*> (weight_ptr), sizeof(double) * weights_[i].n_elem);
    file.write(reinterpret_cast<const char*> (bias_ptr), sizeof(double) * biases_[i].n_elem);
  }
  file.close();
}
