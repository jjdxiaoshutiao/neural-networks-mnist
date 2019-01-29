#include <vector>
#include <armadillo>

#ifndef _NEURAL_NETWORK_H_
#define _NEURAL_NETWORK_H_

class Network {
public:
  Network(const std::vector<uint> &sizes);
  Network(const std::string &filename);
  
  inline int num_layers() const { return num_layers_; };
  inline const std::vector<arma::vec>& biases() const { return biases_; };
  inline const std::vector<arma::mat>& weights() const { return weights_; };
  
  const arma::vec feedforward(arma::vec inputs) const;
  void SGD(std::vector<std::pair<arma::vec, arma::vec>> &training_data, 
           uint epochs, 
           uint mini_batch_size, 
           double eta, 
           const std::vector<std::pair<arma::vec, arma::vec>> &test_data={});
           
  uint evaluate(const std::vector<std::pair<arma::vec, arma::vec>> &test_data) const;
  void save(const std::string &filename) const;
  
private:
  int num_layers_;
  std::vector<uint> sizes_;
  std::vector<arma::vec> biases_;
  std::vector<arma::mat> weights_;  
  
  void update_mini_batch(const std::vector<std::pair<arma::vec, arma::vec>> &mini_batch, double eta);
  std::pair<std::vector<arma::vec>, std::vector<arma::mat>> backpropagation(const arma::vec &x, const arma::vec &y) const;
  inline arma::vec cost_derivative(const arma::vec &output_activitions, const arma::vec &y) const { return output_activitions - y; };
};

#endif