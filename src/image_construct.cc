#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void generate_images(const string &image_file, const string &label_file, const string &output_folder, const int &num_instance) {
  ifstream images(image_file, ios_base::in | ios_base::binary);
  ifstream labels(label_file, ios_base::in | ios_base::binary);
  images.seekg(16);
  labels.seekg(8);
  Mat digit(28, 28, CV_8UC1);
  uchar label;
  for (int i = 0; i < num_instance; ++i) {
    for (int row = 0; row < 28; ++row) {
      for (int col = 0; col < 28; ++col) {
        uchar pixel;
        images.read(reinterpret_cast<char*> (&pixel), sizeof(pixel));
        uchar *ptr = digit.ptr(row, col);
        *ptr = 255 - pixel;
      }
    }
    labels.read(reinterpret_cast<char*> (&label), sizeof(label));
    cout << "Generating image #" << i + 1 << endl;
    imwrite(format("../image/%s/#%i_%i.png", output_folder.c_str(), i + 1, label), digit);
  }

  images.close();
  labels.close();
}

int main(int argc, char *argv[]) {
  generate_images("../data/train-images-idx3-ubyte", "../data/train-labels-idx1-ubyte", "training_set", 60000);
  generate_images("../data/t10k-images-idx3-ubyte", "../data/t10k-labels-idx1-ubyte", "test_set", 10000);

  return 0;
}