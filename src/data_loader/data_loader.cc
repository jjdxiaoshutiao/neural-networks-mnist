#include <fstream>
#include <vector>
#include <utility>

#include <armadillo>

#include "data_loader.h"

using namespace std;
using namespace arma;

void load_data(const string &image_file, const string &label_file, vector<pair<vec, vec>> &data) {
  fstream images(image_file, ios_base::in | ios_base::binary);
  fstream labels(label_file, ios_base::in | ios_base::binary);
  
  images.seekg(16);
  labels.seekg(4);
  int32_t num_instances;
  labels.read(reinterpret_cast<char*> (&num_instances), sizeof(num_instances));
  num_instances = __builtin_bswap32(num_instances);
  unsigned char buffer;
  vector<double> image;
  pair<vec, vec> instance;
  for (uint i = 0; i != num_instances; ++i) {
    for (uint row = 0; row != 28; ++row) {
      for (uint col = 0; col != 28; ++col) {
        images.read(reinterpret_cast<char*> (&buffer), sizeof(buffer));
        image.push_back(buffer / 255.0);
      }
    }
    vec input(image);
    labels.read(reinterpret_cast<char*> (&buffer), sizeof(buffer));
    vec output = zeros<vec>(10);
    output[buffer] = 1;
    instance = make_pair(input, output);
    data.push_back(instance);
    image.clear();
  }
  
  images.close();
  labels.close();
}
