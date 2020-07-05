#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/types_c.h>
#include <iostream>
#include <fstream>
#include <ctime>
# define PI   3.14159265358979323846
using namespace std;
using namespace cv;
// GLobal variables
#define bin_size 20
#define tot_ang 180
#define cellSize 8
struct label{
	cv::Mat feature;
	double distance;
	int label; // 1: cat; 2: dog ; 3: chicken; 
};

bool IsNumber(double x)
{
	return (x == x);
}

cv::Mat CropImage(cv::Mat img) {
	int start_x, start_y, end_x, end_y;
	bool check = true;
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			int x = (int)img.at<uchar>(i, j);
			if (!(x == 255)) {
				start_x = i;
				check = false;
			}
			if (!check) {
				break;
			}
		}
		if (!check) {
			break;
		}
	}

	check = true;
	for (int j = 0; j < img.cols; j++) {
		for (int i = 0; i < img.rows; i++) {
			int x = (int)img.at<uchar>(i, j);
			if (!(x == 255)) {
				start_y = j;
				check = false;
			}
			if (!check) {
				break;
			}
		}
		if (!check) {
			break;
		}
	}

	check = true;
	for (int i = img.rows - 1; i >= 0; i--) {
		for (int j = img.cols - 1; j >= 0; j--) {
			int x = (int)img.at<uchar>(i, j);
			if (!(x == 255)) {
				end_x = i;
				check = false;
			}
			if (!check) {
				break;
			}
		}
		if (!check) {
			break;
		}
	}

	check = true;
	for (int j = img.cols - 1; j >= 0; j--) {
		for (int i = img.rows - 1; i >= 0; i--) {
			int x = (int)img.at<uchar>(i, j);
			if (!(x == 255)) {
				end_y = j;
				check = false;
			}
			if (!check) {
				break;
			}
		}
		if (!check) {
			break;
		}
	}
	cv::Mat ROI(img, cv::Rect(start_y, start_x, end_y - start_y, end_x - start_x));
	cv::Mat croppedImage;
	// Copy the data into new matrix
	ROI.copyTo(croppedImage);
	return croppedImage;
}

cv::Mat GetSquareImage(const cv::Mat& img, int target_width = 512)
{
	int width = img.cols;
	int	height = img.rows;
	cv::Mat square = cv::Mat(target_width, target_width, CV_8UC1, cv::Scalar(255));	
	int max_dim = (width >= height) ? width : height;
	float scale = ((float)target_width) / max_dim;
	cv::Rect roi;
	if (width >= height)
	{
		roi.width = target_width;
		roi.x = 0;
		roi.height = height * scale;
		roi.y = (target_width - roi.height) / 2;
	}
	else
	{
		roi.y = 0;
		roi.height = target_width;
		roi.width = width * scale;
		roi.x = (target_width - roi.width) / 2;
	}
	cv::resize(img, square(roi), roi.size());
	return square;
}

cv::Mat hand(string src) {
	cv::Mat origin_img, img, img_pad, img_d;
	int start_i, end_i, start_j, end_j;

	// doc anh
	origin_img = imread(src, CV_LOAD_IMAGE_GRAYSCALE);
	if (!origin_img.data)
	{
		cout << "Could not open the image" << endl;
	}
	//crop anh, loai bo nhieu nhat nen trang
	cv::Mat img_cropped = CropImage(origin_img);
	//resize ve kich thuoc 512x512
	img = GetSquareImage(img_cropped, 512);

	// chuyen kieu du lieu unsigned integer sang double de de truy cap den phan tu
	img.convertTo(img_d, CV_64FC1);
	// tao vien cho anh
	cv::copyMakeBorder(img_d, img_pad, 1, 1, 1, 1, BORDER_CONSTANT, 255);

	// Tinh toan gradient
	cv::Mat dx = Mat::zeros(img_pad.rows - 2, img_pad.cols - 2, CV_64FC1);
	cv::Mat dy = Mat::zeros(img_pad.rows - 2, img_pad.cols - 2, CV_64FC1);
	cv::Mat dxy = Mat::zeros(img_pad.rows - 2, img_pad.cols - 2, CV_64FC1);
	cv::Mat theta = Mat::zeros(img_pad.rows - 2, img_pad.cols - 2, CV_64FC1);

	for (int i = 1; i < img_pad.rows - 1; i++){
		for (int j = 1; j < img_pad.cols - 1; j++){
			dx.at<double>(i - 1, j - 1) = -1 * img_pad.at<double>(i, j - 1) + img_pad.at<double>(i, j + 1);
			dy.at<double>(i - 1, j - 1) = -1 * img_pad.at<double>(i - 1, j) + img_pad.at<double>(i + 1, j);
			dxy.at<double>(i - 1, j - 1) = sqrt(pow(dx.at<double>(i - 1, j - 1), 2) + pow(dy.at<double>(i - 1, j - 1), 2));
			if (dx.at<double>(i - 1, j - 1) == 0) {
				dx.at<double>(i - 1, j - 1) = 0.000001;
			}
			theta.at<double>(i - 1, j - 1) = atan2(dy.at<double>(i - 1, j - 1), dx.at<double>(i - 1, j - 1)) * (180 / PI);
			if (theta.at<double>(i - 1, j - 1) < 0)
				theta.at<double>(i - 1, j - 1) = theta.at<double>(i - 1, j - 1) + 180;
		}
	}

	// khoi tao ma tran 3 chieu de luu bieu do phan bo do bien thien muc xam
	int row = img.rows;
	int cell_counti = floor(row / cellSize);
	int col = img.cols;
	int cell_countj = floor(col / cellSize);
	int size[3] = { cell_counti, cell_countj, 9 };
	cv::Mat orient_bin(3, size, CV_64FC1, cv::Scalar(0));
	for (int cell_i = 0; cell_i < cell_counti; cell_i++){
		for (int cell_j = 0; cell_j < cell_countj; cell_j++){
			for (int bin = 0; bin < 9; bin++){
				// hang bat dau cua 1 cell
				start_i = (cell_i)*cellSize;
				// hag ket thuc cua 1 cell
				end_i = (cell_i + 1) * cellSize - 1;
				// cot bat dau cua 1 cell
				start_j = (cell_j)*cellSize;
				// cot ket thuc cua 1 cell
				end_j = (cell_j + 1) * cellSize - 1;

				cv::Mat temp = Mat::zeros(end_i - start_i + 1, end_j - start_j + 1, CV_64FC1);

				for (int i = start_i; i <= end_i; i++){
					for (int j = start_j; j <= end_j; j++){		
						if (theta.at<double>(i, j) >= bin*bin_size && theta.at<double>(i, j) < (bin + 1)*bin_size) {
							temp.at<double>(i - start_i, j - start_j) = ((bin + 1) * bin_size - theta.at<double>(i, j)) / bin_size;
						}
						if (bin == 0) {
							if (theta.at<double>(i, j) > (tot_ang - bin_size) && theta.at<double>(i, j) <= tot_ang) {
								temp.at<double>(i - start_i, j - start_j) += (theta.at<double>(i, j) - 160) / bin_size;
							}
						}
						if (bin > 0) {
							if (theta.at<double>(i, j) >= (bin - 1) * bin_size && theta.at<double>(i, j) < bin * bin_size) {
								temp.at<double>(i - start_i, j - start_j) = (theta.at<double>(i, j) - ((bin - 1) * bin_size)) / bin_size;
							}
						}
					}
				}			
				orient_bin.at<double>(cell_i, cell_j, bin) = cv::sum(temp.mul(dxy(Range(start_i, end_i + 1), Range(start_j, end_j + 1))))[0];
			}
		}
	}

	// chuyen 4 cell thanh 1 block
	int cell_block = 4;
	int block_counti = cell_counti - 1;
	int block_countj = cell_countj - 1;

	int size1[2] = { 1,block_counti * block_countj * 36 };
	cv::Mat features(2, size1, CV_64FC1, cv::Scalar(0));
	int fstart = 0;
	int size2[2] = { 1, 9 };
	cv::Mat vect(2, size2, CV_64FC1, cv::Scalar(0));
	for (int block_i = 0; block_i < block_counti; block_i++) {
		for (int block_j = 0; block_j < block_countj; block_j++) {
			cv::Mat block_vect = Mat::zeros(1, 36, CV_64FC1);
			for (int i = 0; i < 2; i++) {
				for (int j = 0; j < 2; j++) {
					int cell_i = block_i + i;
					int  cell_j = block_j + j;
					int cell_b_count = 2 * i + j;
					for (int ii = 0; ii < 9; ii++) {
						block_vect.at<double>(0, ii + cell_b_count * 9) = orient_bin.at<double>(cell_i, cell_j, ii);
					}		
				}
			}

			// chuan hoa vector dac trung
			Mat_<float> norm_block_v;
			double k = 0;
			for (int i = 0; i < 36; i++) {
				k += pow(block_vect.at<double>(0, i), 2);
			}
			k = sqrt(k);
			if (k != 0) {
				norm_block_v = block_vect / k;
			}
			else {
				norm_block_v = block_vect;
			}

			// noi dai cac dac trung cua block (16x16) thanh vector dac trung cua anh
			for (int jj = 0; jj < 36; jj++)
				features.at<double>(0, jj + fstart) = norm_block_v.at<float>(0, jj);
			fstart += 36;
		}
	}
	return features;
}

void writeFeature(string name_file, cv::Mat feature) {
	cv::FileStorage fs(name_file, cv::FileStorage::WRITE);
	fs << "feature" << feature;
}
cv::Mat readFeature(string src) {
	cv::Mat feature;
	cv::FileStorage fs2(src, FileStorage::READ);
	fs2["feature"] >> feature;
	return feature;
}

double calDistanceL2Norm(cv::Mat feature_first, cv::Mat feature_second) {
	double distance = 0;
	for (int i = 0; i < feature_first.cols; i++) {
		distance = distance + pow(feature_first.at<double>(0, i) - feature_second.at<double>(0, i), 2);
	}
	distance = sqrt(distance);
	return distance;
}

double calCosin(cv::Mat feature_first, cv::Mat feature_second) {
	double tu = 0;
	double mau = 0;
	for (int i = 0; i < feature_first.cols; i++) {
		tu = tu + feature_first.at<double>(0, i) * feature_second.at<double>(0, i);
	}
	double temp1 = 0;
	double temp2 = 0;
	for (int i = 0; i < feature_first.cols; i++) {
		temp1 += pow(feature_first.at<double>(0, i), 2);
	}
	for (int i = 0; i < feature_second.cols; i++) {
		temp2 += pow(feature_second.at<double>(0, i), 2);
	}
	mau = sqrt(temp1) * sqrt(temp2);
	return tu / mau;
}

double calDistanceNorm(cv::Mat feature_first, cv::Mat feature_second) {
	double distance = 0;
	for (int i = 0; i < feature_first.cols; i++) {
		distance = distance + abs(feature_first.at<double>(0, i) - feature_second.at<double>(0, i));
	}
	return distance;
}

int main(int argc, char* argv[])
{
	// doc anh
	/*cv::Mat img = imread("ga.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	if (!img.data)
	{
		cout << "Could not open the image" << endl;
	}
	
	cout << img << endl;
	return 0;*/
	////crop anh, loai bo nhieu nhat nen trang
	//cv::Mat img_cropped = CropImage(img);
	////resize ve kich thuoc 512x512
	//cv::Mat img_resize = GetSquareImage(img_cropped, 512);
	//cv::Mat feature1 = hand("train/images/cat.2.jpg");
	//cout << feature1.size() << endl;
	//return 0; 
	/*hand("chicken.11.jpg");
	return 0;
	cv::Mat origin_img = imread("test/images/7.jpg");
	if (!origin_img.data)
	{
		cout << "Could not open the image" << endl;
	}
	//crop anh, loai bo nhieu nhat nen trang
	cv::Mat img_cropped = CropImage(origin_img);
	//resize ve kich thuoc 512x512
	cv::Mat img_resize = GetSquareImage(img_cropped, 128);
	cv::Mat img;
	cv::cvtColor(img_resize, img, cv::COLOR_BGR2GRAY);
	cout << img << endl;
	namedWindow("Display window", WINDOW_AUTOSIZE);// Create a window for display.
	imshow("Display window", img);
	waitKey(0);
	return 0; */
	label trainArray[100];
	label temp;
	int N = 30; // so anh cua moi nhan
	cv:Mat feature;
	cv::Mat feature_test;
	int type;
	string src_test;
	do {
		cout << "0.Thoat" << endl;
		cout << "1. Doc anh tu dataset va ghi dac trung" << endl;
		cout << "2. Test anh" << endl;
		cout << "Nhap lua chon :";
		cin >> type;
		if (type == 1) {
			cout << "Dang tinh dac trung cua anh dau vao và luu vao file...." << endl;
			for (int i = 0; i < N; i++) {
				feature = hand("train/images/cat." + to_string(i + 1) + ".jpg");
				writeFeature("train/features/cat." + to_string(i + 1) + ".yml", feature);
			}
			for (int i = 0; i < N; i++) {
				feature = hand("train/images/dog." + to_string(i + 1) + ".jpg");
				writeFeature("train/features/dog." + to_string(i + 1) + ".yml", feature);
			}
			for (int i = 0; i < N; i++) {
				feature = hand("train/images/chicken." + to_string(i + 1) + ".jpg");
				writeFeature("train/features/chicken." + to_string(i + 1) + ".yml", feature);
			}
			cout << "done" << endl;
		}
		else if (type == 2) {
			do {
				int compare_type;				
				cout << "Nhap file anh test : ";
				cin >> src_test;
				cout << "1. Normal | 2. L2Norm | 3.Cosin" << endl;
				cout << "Chon cong thuc so sanh anh" << endl;
				cin >> compare_type;
				cout << "Dang du doan. Vui long cho!" << endl;
				feature_test = hand("test/images/" + src_test);
				for (int i = 0; i < N; i++) {
					feature = readFeature("train/features/cat." + to_string(i + 1) + ".yml");
					trainArray[i].feature = feature;
					trainArray[i].label = 1;
					if (compare_type == 1) {
						trainArray[i].distance = calDistanceNorm(feature, feature_test);
					}
					else if (compare_type == 2) {
						trainArray[i].distance = calDistanceL2Norm(feature, feature_test);
					}
					else if (compare_type == 3) {
						trainArray[i].distance = calCosin(feature, feature_test);
					}
				}
				for (int i = 0; i < N; i++) {
					feature = readFeature("train/features/dog." + to_string(i + 1) + ".yml");
					trainArray[i + N].feature = feature;
					trainArray[i + N].label = 2;
					if (compare_type == 1) {
						trainArray[i + N].distance = calDistanceNorm(feature, feature_test);
					}
					else if (compare_type == 2) {
						trainArray[i + N].distance = calDistanceL2Norm(feature, feature_test);
					}
					else if (compare_type == 3) {
						trainArray[i + N].distance = calCosin(feature, feature_test);
					}
				}
				for (int i = 0; i < N; i++) {
					feature = readFeature("train/features/chicken." + to_string(i + 1) + ".yml");
					trainArray[i + 2 * N].feature = feature;
					trainArray[i + 2 * N].label = 3;
					if (compare_type == 1) {
						trainArray[i + 2 * N].distance = calDistanceNorm(feature, feature_test);
					}
					else if (compare_type == 2) {
						trainArray[i + 2 * N].distance = calDistanceL2Norm(feature, feature_test);
					}
					else if (compare_type == 3) {
						trainArray[i + 2 * N].distance = calCosin(feature, feature_test);
					}
				}
				if (compare_type == 1 || compare_type == 2) {
					for (int i = 0; i < 3 * N - 1; i++) {
						for (int j = i + 1; j < 3 * N; j++) {
							if (trainArray[i].distance > trainArray[j].distance) {
								temp = trainArray[i];
								trainArray[i] = trainArray[j];
								trainArray[j] = temp;
							}
						}
					}
				}
				else if (compare_type == 3) {
					for (int i = 0; i < 3 * N - 1; i++) {
						for (int j = i + 1; j < 3 * N; j++) {
							if (trainArray[i].distance < trainArray[j].distance) {
								temp = trainArray[i];
								trainArray[i] = trainArray[j];
								trainArray[j] = temp;
							}
						}
					}
				}

				/*for (int i = 0; i < 30; i++) {
					cout << trainArray[i].distance << "   " << trainArray[i].label << endl;
				}*/

				int cat_predict = 0;
				int dog_predict = 0;
				int chicken_predict = 0;
				for (int i = 0; i < 30; i++) {
					if (trainArray[i].label == 1) {
						cat_predict++;
					}
					else if (trainArray[i].label == 2) {
						dog_predict++;
					}
					else if (trainArray[i].label == 3) {
						chicken_predict++;
					}
				}
				if (cat_predict >= dog_predict && cat_predict >= chicken_predict) {
					cout << "Du doan la meo" << endl;
				}
				else if (dog_predict >= cat_predict && dog_predict >= chicken_predict) {
					cout << "Du doan la cho" << endl;
				}
				else if (chicken_predict >= cat_predict && chicken_predict >= dog_predict) {
					cout << "Du doan la ga" << endl;
				}
				cout << "Nhap lua chon :";
				cin >> type;				
			} while (type == 2);
		}
	} while (type != 0);
	return 0;
}