#include <armadillo>

#include "sigmoid.h"

using namespace arma;

vec sigmoid(const vec &z) {
  return pow(ones<vec>(z.n_elem) + exp(-z), -1);
}

vec sigmoid_prime(const vec &z) {
  return sigmoid(z) % (ones<vec>(z.n_elem) - sigmoid(z));
}