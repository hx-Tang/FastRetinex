#include <iostream>
#include <algorithm>
#include <ctime>
#include <omp.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace std;
using namespace cv;


//log2计算表
void log2dic(double *Log2);
//mat转数组
void mat2arr(Mat, double *, uchar *);
//数组转mat
Mat arr2mat(double *,Mat);
//计算递归系数
void Calc(int sigma, double &b0, double &b1, double &b2, double &b3);
//递归高斯滤波
void filterGaussian(double *tmp, int cols, int rows, double b0, double b1, double b2, double b3);
//单尺度
void SSR(uchar * usrc, double *src,double *res, int sigma,int rows, int cols, double *Log2);
//多尺度
void MSR(double *res, double *res1, double *res2, int size);
//色彩恢复
void MSRCR(uchar *src, double *res,int rows, int cols, double *);


/*
	改进说明：基于fastRetinex.cpp
	1. 编译器设置优化
	2. 递归高斯滤波算法
	3. openmp并行加速
	4. 历遍指针计算移动到循环外

	效果：提速5倍左右
*/

void main() 
{
	//预置log2运算供查表 不计时
	double * dic = new double[256*192];
	log2dic(dic);
	
	//计时
	time_t start, stop;
	start = clock();

	//读RGB图
	Mat img = imread("test10.jpg");

	int size = img.rows * img.cols * 3;

	//拆分通道丢进三维数组
	uchar *src = new uchar[size];
	double *BGR = new double[size];
	mat2arr(img, BGR, src);
	
	//SSR * 3
	double *res = new double[size];
	SSR(src, BGR, res, 80, img.rows, img.cols, dic);

	double *res1 = new double[size];
	SSR(src, BGR, res, 15, img.rows, img.cols, dic);

	double *res2 = new double[size];
	SSR(src, BGR, res, 200, img.rows, img.cols, dic);

	//Mat img_ssr = arr2mat(res, img);

	//MSR
	MSR(res, res1, res2, img.rows*img.cols);

	//Mat img_msr = arr2mat(res, img);

	//MSRCR
	MSRCR(src, res, img.rows, img.cols, dic);

	Mat img_msrcr = arr2mat(res, img);

	stop = clock();

	cout << "Use Time:" << (double)(stop - start) / CLOCKS_PER_SEC << endl;

	//展示
	//imshow("原图", img);
	//imshow("单尺度Retinex", img_ssr);
	//imshow("多尺度Retinex", img_msr);
	imshow("多尺度Retinex带色彩复原", img_msrcr);

	waitKey(0);
}

//double、大int、小int互转
inline int int2smallint(int x) { return (x >> 10); }
inline int double2int(double x) { return (int)(x * 1024 + 0.5); }

//预置log2运算表
void log2dic(double * dic)
{
	#pragma omp parallel for
	for (int i = 0; i < 256 * 192; i++) {
		dic[i] = log2(i);
	}
}

//拆mat
void mat2arr(Mat img, double *BGR, uchar *src)
{
	#pragma omp parallel for
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			for (int k = 0; k < 3; k++)
			{
				BGR[i * img.cols * 3 + j * 3 + k] = (double)img.at<Vec3b>(i, j)[k];
				src[i * img.cols * 3 + j * 3 + k] = (uchar)img.at<Vec3b>(i, j)[k];
			}
		}
	}
}

//组装mat
Mat arr2mat(double *BGR, Mat src)
{
	double max = *max_element(BGR, (BGR + src.rows * src.cols * 3));
	double min = *min_element(BGR, (BGR + src.rows * src.cols * 3));

	Mat img;
	src.copyTo(img);

	#pragma omp parallel for
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			for (int k = 0; k < 3; k++)
			{
				//归一化
				img.at<Vec3b>(i, j)[k] = (uchar)(255 * (BGR[i * img.cols * 3 + j * 3 + k] - min) / (max - min));
			}
		}
	}
	return img;
}

//SSR
void SSR(uchar *usrc, double *src, double *res, int sigma,int rows,int cols, double* Log2)
{
	//生成递归系数
	double b0, b1, b2, b3;
	Calc(sigma, b0, b1, b2, b3);
	
	//中间变量
	double *tmp = new double[(rows+6) * cols * 3];
	//上下各留三行方便滤波
	memcpy(tmp + 3 * cols * 3, src, rows * cols * 3 * sizeof(double));

	//滤波
	filterGaussian(tmp,cols, rows, b0,b1,b2,b3);

	//对数相减 转回实数域
	for (int i = 0; i < rows * cols * 3; i++)
	{
		res[i] = pow(2, (Log2[usrc[i]] - Log2[(uchar)tmp[i]]));
		//res[i] = tmp[i];
	}

}

//MSR
void MSR(double *res, double *res1, double * res2, int size)
{
	//加权平均
	for (int i = 0; i < size * 3; i++)
	{
		res[i] = 0.33*res1[i] + 0.34*res[i] + 0.33*res2[i];
	}
}

//MSRCR
void MSRCR(uchar *src, double *res , int rows, int cols, double * Log2)
{
	//CR参数
	int G = 192;
	int O = -30;
	int alpha = 125;
	int beta = 46;

	//计算CR

	#pragma omp parallel for
	for (int i = 0; i < rows; i++)
	{
		uchar *pS = src + i * cols * 3;
		double *pr = res + i * cols * 3;
		for (int j = 0; j < cols; j++, pr += 3, pS+=3)
		{
			pr[0] = G * ((beta * (Log2[alpha*pS[0]] - Log2[pS[0] + pS[1] + pS[2]])) * pr[0] + O);
			pr[1] = G * ((beta * (Log2[alpha*pS[1]] - Log2[pS[0] + pS[1] + pS[2]])) * pr[1] + O);
			pr[2] = G * ((beta * (Log2[alpha*pS[2]] - Log2[pS[0] + pS[1] + pS[2]])) * pr[2] + O);
		}
	}
}

/*
	递归高斯滤波
*/

//计算递归系数b0，b1，b2，b3
void Calc(int sigma, double &b0, double &b1, double &b2, double &b3)
{
	//确定卷积核半径 
	//几次尝试后 1/4倍sigma 效果比较好 
	double Radius = sigma/4;

	//论文里的算法
	double Q, B, B1, B2, B3;
	Q = (double)(0.98711 * Radius - 0.96330);
	B = 1.57825 + 2.44413 * Q + 1.4281 * Q * Q + 0.422205 * Q * Q * Q;
	B1 = 2.44413 * Q + 2.85619 * Q * Q + 1.26661 * Q * Q * Q;
	B2 = -1.4281 * Q * Q - 1.26661 * Q * Q * Q;
	B3 = 0.422205 * Q * Q * Q;

	//归一化
	b0 = 1.0 - (B1 + B2 + B3) / B;
	b1 = B1 / B;
	b2 = B2 / B;
	b3 = B3 / B;
}

//二维递归高斯滤波 启用多线程
void filterGaussian(double *tmp, int cols, int rows, double b0, double b1, double b2, double b3)
{

	//X轴滤波

	//从左到右
	#pragma omp parallel for
	for (int i = 0; i < rows; i++)
	{
		// 从左往右读像素
		double *prL = (tmp + 3 * cols * 3) + i * cols * 3;

		// 边缘处重复像素
		double B1 = prL[0], B2 = prL[0], B3 = prL[0];
		double G1 = prL[1], G2 = prL[1], G3 = prL[1];
		double R1 = prL[2], R2 = prL[2], R3 = prL[2];
		for (int j = 0; j < cols; j++, prL += 3)
		{
			// 顺向迭代
			prL[0] = prL[0] * b0 + B1 * b1 + B2 * b2 + B3 * b3;
			prL[1] = prL[1] * b0 + G1 * b1 + G2 * b2 + G3 * b3;
			prL[2] = prL[2] * b0 + R1 * b1 + R2 * b2 + R3 * b3;
			// 前向移位
			B3 = B2, B2 = B1, B1 = prL[0];
			G3 = G2, G2 = G1, G1 = prL[1];
			R3 = R2, R2 = R1, R1 = prL[2];
		}
		// 从右往左 
		double *prR = (tmp + 3 * cols * 3) + (i + 1) * cols * 3 - 3;

		// 边缘处重复像素
		double B1R = prR[0], B2R = prR[0], B3R = prR[0];
		double G1R = prR[1], G2R = prR[1], G3R = prR[1];
		double R1R = prR[2], R2R = prR[2], R3R = prR[2];
		for (int j = 0; j < cols; j++, prR -= 3)
		{
			// 反向迭代
			prR[0] = prR[0] * b0 + B1R * b1 + B2R * b2 + B3R * b3;
			prR[1] = prR[1] * b0 + G1R * b1 + G2R * b2 + G3R * b3;
			prR[2] = prR[2] * b0 + R1R * b1 + R2R * b2 + R3R * b3;

			// 后向移位
			B3R = B2R, B2R = B1R, B1R = prR[0];
			G3R = G2R, G2R = G1R, G1R = prR[1];
			R3R = R2R, R2R = R1R, R1R = prR[2];
		}
	}

	//Y轴滤波

	//复制头三行BGR像素
	memcpy(tmp + 0 * cols * 3, tmp + 3 * cols * 3, cols * 3 * sizeof(double));
	memcpy(tmp + 1 * cols * 3, tmp + 3 * cols * 3, cols * 3 * sizeof(double));
	memcpy(tmp + 2 * cols * 3, tmp + 3 * cols * 3, cols * 3 * sizeof(double));
	//复制尾三行BGR像素
	memcpy(tmp + (rows + 3) * cols * 3, tmp + (rows + 2) * cols * 3, cols * 3 * sizeof(double));
	memcpy(tmp + (rows + 4) * cols * 3, tmp + (rows + 2) * cols * 3, cols * 3 * sizeof(double));
	memcpy(tmp + (rows + 5) * cols * 3, tmp + (rows + 2) * cols * 3, cols * 3 * sizeof(double));

	//列向下迭代
	for (int i = 0; i < rows; i++)
	{
		//设定起始值
		double *pr3 = tmp + (i + 0) * cols * 3;
		double *pr2 = tmp + (i + 1) * cols * 3;
		double *pr1 = tmp + (i + 2) * cols * 3;
		double *pr0 = tmp + (i + 3) * cols * 3;
		//从左往右做完一行的像素
		for (int j = 0; j < cols; j++, pr0 += 3, pr1 += 3, pr2 += 3, pr3 += 3)
		{
			// 顺向迭代
			pr0[0] = pr0[0] * b0 + pr1[0] * b1 + pr2[0] * b2 + pr3[0] * b3;
			pr0[1] = pr0[1] * b0 + pr1[1] * b1 + pr2[1] * b2 + pr3[1] * b3;
			pr0[2] = pr0[2] * b0 + pr1[2] * b1 + pr2[2] * b2 + pr3[2] * b3;
		}
	}

	//列向上迭代
	for (int i = rows - 1; i >= 0; i--)
	{
		// 设定起始值
		double *pr3 = tmp + (i + 3) * cols * 3;
		double *pr2 = tmp + (i + 2) * cols * 3;
		double *pr1 = tmp + (i + 1) * cols * 3;
		double *pr0 = tmp + (i + 0) * cols * 3;
		//从左往右做完一行的像素
		for (int j = 0; j < cols; j++, pr0 += 3, pr1 += 3, pr2 += 3, pr3 += 3)
		{
			// 反向迭代
			pr0[0] = pr0[0] * b0 + pr1[0] * b1 + pr2[0] * b2 + pr3[0] * b3;
			pr0[1] = pr0[1] * b0 + pr1[1] * b1 + pr2[1] * b2 + pr3[1] * b3;
			pr0[2] = pr0[2] * b0 + pr1[2] * b1 + pr2[2] * b2 + pr3[2] * b3;
		}
	}
}