#include <iostream>
#include <algorithm>
#include <time.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

double * mat2arr(Mat);
Mat arr2mat(double *, Mat);
double * SSR(int, double *, Mat);
double * MSR(double *, Mat);
double * MSRCR(double *, Mat);
double * createFilter(int, Mat);
double * filterGaussian(double *, int, double *, Mat);


int main()
{
	time_t start, end;
	start = clock();
	//读RGB图
	Mat img = imread("test6.jpg");
	imshow("原图", img);
	
	//拆分通道丢进三维数组
	double *BGR = new double[img.rows * img.cols * 3];
	BGR = mat2arr(img);

	//ssr msr 与 msrcr 
	Mat img_ssr = arr2mat(SSR(80,BGR,img),img);
	Mat img_msr = arr2mat(MSR(BGR,img),img);
	Mat img_msrcr = arr2mat(MSRCR(BGR,img),img);

	end = clock();
	cout << "Use Time:" << (double)(end - start) / CLOCKS_PER_SEC << endl;

	//展示
	imshow("单尺度Retinex", img_ssr);
	imshow("多尺度Retinex", img_msr);
	imshow("多尺度Retinex带色彩复原", img_msrcr);	

	waitKey(0);
	
	return 0;
}

//拆mat
double * mat2arr(Mat img)
{

	double *BGR = new double[img.rows * img.cols * 3];

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			for (int k = 0; k < 3; k++)
			{
				BGR[i * img.cols * 3 + j * 3 + k] = (double)img.at<Vec3b>(i, j)[k];
			}
		}
	}
	return BGR;
}


//组装mat
Mat arr2mat(double *BGR, Mat src)
{
	double max = *max_element(BGR,(BGR + src.rows * src.cols * 3));
	double min = *min_element(BGR,(BGR + src.rows * src.cols * 3));
	
	Mat img;
	src.copyTo(img);

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			for (int k = 0; k < 3; k++)
			{
				//归一化
				img.at<Vec3b>(i, j)[k] = (uchar)(255 * (BGR[i * img.cols * 3 + j * 3 + k]-min)/(max-min));
			}
		}
	}
	return img;
}


//SSR
double * SSR(int sigma, double *BGR, Mat img)
{
	double *res = new double[img.rows * img.cols * 3];

	//高斯滤波器
	double *filter1 = createFilter(sigma,img);

	//滤波
	double *BGR1 = filterGaussian(filter1, sigma, BGR, img);

	//对数相减 转回实数域
	for (int i = 0; i < img.rows * img.cols * 3;i++)
	{
		res[i] = pow(2,(log2(BGR[i]) - log2(BGR1[i])));	
	}

	return res;
}


//MSR
double * MSR(double *BGR, Mat img)
{
	double *res = new double[img.rows * img.cols * 3];
	//三阶MSR
	int sigma1 = 15;
	int sigma2 = 80;
	int sigma3 = 250;
	double *BGR1 = SSR(sigma1, BGR, img);
	double *BGR2 = SSR(sigma2, BGR, img);
	double *BGR3 = SSR(sigma3, BGR, img);

	//加权平均
	for (int i = 0; i < img.rows * img.cols * 3; i++)
	{
		res[i] = 0.33*BGR1[i] + 0.34*BGR2[i] + 0.33*BGR3[i];
	}
	
	return res;
}

//MSRCR
double * MSRCR(double *BGR, Mat img)
{
	double *res = new double[img.rows * img.cols * 3];
	//CR参数
	double G = 192;
	double O = -30;
	double alpha = 125;
	double beta = 46;
	
	//计算CR
	double *CR = new double[img.rows * img.cols * 3];

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			CR[i * img.cols * 3 + j * 3 + 0] = beta * (log10(alpha*BGR[i * img.cols * 3 + j * 3 + 0])
				- log10(BGR[i * img.cols * 3 + j * 3 + 0] + BGR[i * img.cols * 3 + j * 3 + 1] + BGR[i * img.cols * 3 + j * 3 + 2]));
			CR[i * img.cols * 3 + j * 3 + 1] = beta * (log10(alpha*BGR[i * img.cols * 3 + j * 3 + 1])
				- log10(BGR[i * img.cols * 3 + j * 3 + 0] + BGR[i * img.cols * 3 + j * 3 + 1] + BGR[i * img.cols * 3 + j * 3 + 2]));
			CR[i * img.cols * 3 + j * 3 + 2] = beta * (log10(alpha*BGR[i * img.cols * 3 + j * 3 + 2])
				- log10(BGR[i * img.cols * 3 + j * 3 + 0] + BGR[i * img.cols * 3 + j * 3 + 1] + BGR[i * img.cols * 3 + j * 3 + 2]));
		}
	}
	//MSR滤波
	res = MSR(BGR, img);

	//修正
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			res[i * img.cols * 3 + j * 3 + 0] = G * (CR[i * img.cols * 3 + j * 3 + 0] * res[i * img.cols * 3 + j * 3 + 0] + O);
			res[i * img.cols * 3 + j * 3 + 1] = G * (CR[i * img.cols * 3 + j * 3 + 1] * res[i * img.cols * 3 + j * 3 + 1] + O);
			res[i * img.cols * 3 + j * 3 + 2] = G * (CR[i * img.cols * 3 + j * 3 + 2] * res[i * img.cols * 3 + j * 3 + 2] + O);
		}
	}
	return res;
}

//二维高斯滤波
double * filterGaussian(double * filter, int sigma, double * BGR, Mat img)
{
	int size = sigma + 1;
	double *tmp = new double[img.rows * img.cols * 3];
	double *res = new double[img.rows * img.cols * 3];
	//cout << size << img.cols << img.rows << endl;
	//x轴滤波
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			//卷积
			for (int k = 0; k < size; k++)
			{
				int p = i + sigma/2 + 1 - k;

				if (p > img.rows - 1) p = 2 * img.rows -1 - p;
				if (p < 0) p = p * -1;
				tmp[i * img.cols * 3 + j * 3 + 0] += filter[k] * BGR[p * img.cols * 3 + j * 3 + 0];
				tmp[i * img.cols * 3 + j * 3 + 1] += filter[k] * BGR[p * img.cols * 3 + j * 3 + 1];
				tmp[i * img.cols * 3 + j * 3 + 2] += filter[k] * BGR[p * img.cols * 3 + j * 3 + 2];
			}
		}	
	}
	//y轴滤波
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			//卷积
			for (int k = 0; k < size; k++)
			{
				int p = j + size / 2 - k;
				if (p < 0) p *= -1;
				if (p > img.cols - 1) p = 2 * (img.cols - 1) - p;
				res[i * img.cols * 3 + j * 3 + 0] += filter[k] * tmp[i * img.cols * 3 + p * 3 + 0];
				res[i * img.cols * 3 + j * 3 + 1] += filter[k] * tmp[i * img.cols * 3 + p * 3 + 1];
				res[i * img.cols * 3 + j * 3 + 2] += filter[k] * tmp[i * img.cols * 3 + p * 3 + 2];
			}
		}
	}

	return res;
}

//生成一维高斯核
double * createFilter(int sigma, Mat img)
{
	//分配大小
	int size = sigma + 1;
	double *filter = new double[size];

	// 计算
	double sum = 0;
	for (int i = 0; i < size; i++)
	{
		double x = i - (size / 2);
		filter[i] = exp(-(x * x) / (2 * sigma * sigma));
		sum += filter[i];
	}
	//归一化
	for (int i = 0; i < size; i++) {
		filter[i] /= sum;
	}
	return filter;
}
