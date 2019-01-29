#include <vector>
#include <utility>

#include <armadillo>

#ifndef _DATA_LOADER_H_
#define _DATA_LOADER_H_

void load_data(const std::string &image_file, const std::string &label_file, std::vector<std::pair<arma::vec, arma::vec>> &data);

#endif