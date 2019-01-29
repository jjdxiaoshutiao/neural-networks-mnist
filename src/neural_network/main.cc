#include <vector>

#include <armadillo>

#include "../data_loader/data_loader.h"
#include "neural_network.h"


using namespace std;
using namespace arma;

int main(int argc, char *argv[]) {
  vector<pair<vec, vec>> training_data;
  vector<pair<vec, vec>> test_data;
  vector<pair<vec, vec>> validation_data;
  cout << "Loading mnist handwritten digit data..." << endl;
  load_data("../../data/train-images-idx3-ubyte", "../../data/train-labels-idx1-ubyte", training_data);
  load_data("../../data/t10k-images-idx3-ubyte", "../../data/t10k-labels-idx1-ubyte", test_data);
  for (size_t i = 0; i != 10000; ++i) {
    validation_data.push_back(training_data.back());
    training_data.pop_back();
  }
  cout << "Finish loading data." << endl;
  
  vector<uint> sizes = { 784, 30, 10 };
  Network net_1(sizes);
  cout << "Training..." << endl;
  net_1.SGD(training_data, 30, 10, 3.0, validation_data);
  net_1.save("net_data");
  
  Network net_2("net_data");
  cout << "Evaluating network with test data..." << endl;
  cout << "Test data evaluation accuracy: " << net_2.evaluate(test_data) << "/10000" << endl;
  
  return 0;
}