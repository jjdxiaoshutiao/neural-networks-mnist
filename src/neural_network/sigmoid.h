#include <armadillo>

#ifndef _SIGMOID_H_
#define _SIGMOID_H_

arma::vec sigmoid(const arma::vec &z);
arma::vec sigmoid_prime(const arma::vec &z);

#endif