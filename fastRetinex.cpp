#include <iostream>
#include <algorithm>
#include <ctime>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace std;
using namespace cv;

void logdic(double*);
void mat2arr(Mat, uchar *);
Mat arr2mat(double *,Mat);
int * createFilter(int);//sigma
uchar * filterGaussian(uchar *, int *, int, int, int);//imgָ�롢filterָ�롢sigma��cols
void SSR(uchar *,double *, int,int, int, double *);//imgָ�롢resָ�롢sigma��cols��rows��log��
void MSR(double *, double*, double*, int);//����ssr��size
void MSRCR(uchar *, double *,int, int, double *);//imgָ�롢resָ�롢sigma��cols��rows


/*
	�Ľ�˵����
	1. �������̡�����SSR MSR ���
	2. ���ٲ���Ҫ�����鿽�����������
	3. ʹ�ý��Ƹ�������ͣ�ԭֵ*1024���˲���
	4. ʹ��ά��˹�˲���Ϊ����һά�������㣬�����������10bit��ԭ
	5. Ԥ��log2�����ȡ��SSR��CR�е�log����
	6. ʹ��inline�ؼ���

	Ч��������10��

	��һ�������滻�µĿ����˲��㷨
*/

void main() 
{
	//Ԥ��log2���㹩��� ����ʱ
	double * dic = new double[256*192];
	logdic(dic);
	
	//��ʱ
	time_t start, stop;
	start = clock();

	//��RGBͼ
	Mat img = imread("T.png");

	//���ͨ��������ά����
	uchar *BGR = new uchar[img.rows * img.cols * 3];
	mat2arr(img, BGR);
	
	//SSR * 3
	double *res = new double[img.rows * img.cols * 3];
	SSR(BGR, res, 80, img.rows, img.cols, dic);

	double *res1 = new double[img.rows * img.cols * 3];
	SSR(BGR, res1, 15, img.rows, img.cols, dic);

	double *res2 = new double[img.rows * img.cols * 3];
	SSR(BGR, res2, 200, img.rows, img.cols, dic);

	Mat img_ssr = arr2mat(res, img);

	//MSR
	MSR(res, res1, res2, img.rows*img.cols);

	Mat img_msr = arr2mat(res, img);

	//MSRCR
	MSRCR(BGR, res, img.rows, img.cols, dic);

	Mat img_msrcr = arr2mat(res, img);

	stop = clock();

	cout << "Use Time:" << (double)(stop - start) / CLOCKS_PER_SEC << endl;

	//չʾ
	//imshow("ԭͼ", img);
	//imshow("���߶�Retinex", img_ssr);
	//imshow("��߶�Retinex", img_msr);
	//imshow("��߶�Retinex��ɫ�ʸ�ԭ", img_msrcr);
	//waitKey(0);
	imwrite("Te.jpg", img_msrcr);
}

//double����int��Сint��ת
inline int int2smallint(int x) { return (x >> 10); }
inline int double2int(double x) { return (int)(x * 1024 + 0.5); }

//Ԥ��log2�����
void logdic(double * dic)
{
	for (int i = 0; i < 256 * 192; i++) {
		dic[i] = log2(i);
	}
}

//��mat
void mat2arr(Mat img, uchar *BGR)
{
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			for (int k = 0; k < 3; k++)
			{
				BGR[i * img.cols * 3 + j * 3 + k] = (uchar)img.at<Vec3b>(i, j)[k];
			}
		}
	}
}

//��װmat
Mat arr2mat(double *BGR, Mat src)
{
	double max = *max_element(BGR, (BGR + src.rows * src.cols * 3));
	double min = *min_element(BGR, (BGR + src.rows * src.cols * 3));

	Mat img;
	src.copyTo(img);

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			for (int k = 0; k < 3; k++)
			{
				//��һ��
				img.at<Vec3b>(i, j)[k] = (uchar)(255 * (BGR[i * img.cols * 3 + j * 3 + k] - min) / (max - min));
			}
		}
	}
	return img;
}

//SSR
void SSR(uchar *BGR, double *res, int sigma,int rows,int cols, double* Log2)
{

	//���ɸ�˹�˲���
	int *filter = createFilter(sigma);

	//�˲�
	uchar *BGR1 = filterGaussian(BGR, filter, sigma, rows, cols);

	//������� ת��ʵ����
	for (int i = 0; i < rows * cols * 3; i++)
	{
		res[i] = pow(2, (Log2[(int)BGR[i]] - Log2[(int)BGR1[i]]));
	}

}

//MSR
void MSR(double *res, double *res1, double * res2, int size)
{
	//��Ȩƽ��
	for (int i = 0; i < size * 3; i++)
	{
		res[i] = 0.33*res1[i] + 0.34*res[i] + 0.33*res2[i];
	}
}

//MSRCR
void MSRCR(uchar *BGR, double *res , int rows, int cols, double * Log2)
{
	//CR����
	int G = 192;
	int O = -30;
	int alpha = 125;
	int beta = 46;

	//����CR
	double *CR = new double[rows * cols * 3];

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			CR[i * cols * 3 + j * 3 + 0] = beta * (Log2[alpha*BGR[i * cols * 3 + j * 3 + 0]]
				- Log2[BGR[i * cols * 3 + j * 3 + 0] + BGR[i * cols * 3 + j * 3 + 1] + BGR[i * cols * 3 + j * 3 + 2]]);
			CR[i * cols * 3 + j * 3 + 1] = beta * (Log2[alpha*BGR[i * cols * 3 + j * 3 + 1]]
				- Log2[BGR[i * cols * 3 + j * 3 + 0] + BGR[i * cols * 3 + j * 3 + 1] + BGR[i * cols * 3 + j * 3 + 2]]);
			CR[i * cols * 3 + j * 3 + 2] = beta * (Log2[alpha*BGR[i * cols * 3 + j * 3 + 2]]
				- Log2[BGR[i * cols * 3 + j * 3 + 0] + BGR[i * cols * 3 + j * 3 + 1] + BGR[i * cols * 3 + j * 3 + 2]]);
		}
	}

	//ɫ������
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			res[i * cols * 3 + j * 3 + 0] = G * (CR[i * cols * 3 + j * 3 + 0] * res[i * cols * 3 + j * 3 + 0] + O);
			res[i * cols * 3 + j * 3 + 1] = G * (CR[i * cols * 3 + j * 3 + 1] * res[i * cols * 3 + j * 3 + 1] + O);
			res[i * cols * 3 + j * 3 + 2] = G * (CR[i * cols * 3 + j * 3 + 2] * res[i * cols * 3 + j * 3 + 2] + O);
		}
	}
}

//����һά��˹�� ��������
int * createFilter(int sigma)
{
	//�����С
	int size = (int)sigma + 1;

	double *tmpfilter = new double[size];
	int *filter = new int[size];

	// ����
	double sum = 0;
	for (int i = 0; i < size; i++)
	{
		double x = i - (size / 2);
		tmpfilter[i] = exp(-(x * x) / (2 * sigma * sigma));
		sum += tmpfilter[i];
	}
	//��һ�� תint
	for (int i = 0; i < size; i++) {
		tmpfilter[i] /= sum;
		filter[i] = double2int(tmpfilter[i]);
	}
	return filter;
}

//��ά��˹�˲�
uchar * filterGaussian(uchar * BGR, int * filter, int sigma,int rows, int cols)
{
	int size = sigma + 1;
	uchar *res = new uchar[rows * cols * 3];
	int v1, v2, v3;

	//x���˲�
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			v1 = v2 = v3 = 0;
			//���
			for (int k = 0; k < size; k++)
			{
				int p = i + sigma / 2 + 1 - k;
				if (p > rows - 1) p = 2 * rows - 1 - p;
				if (p < 0) p = p * -1;

				v1 += filter[k] * BGR[p * cols * 3 + j * 3 + 0];
				v2 += filter[k] * BGR[p * cols * 3 + j * 3 + 1];
				v3 += filter[k] * BGR[p * cols * 3 + j * 3 + 2];
			}
			res[i * cols * 3 + j * 3 + 0] = (uchar)int2smallint(v1);
			res[i * cols * 3 + j * 3 + 1] = (uchar)int2smallint(v2);
			res[i * cols * 3 + j * 3 + 2] = (uchar)int2smallint(v3);
		}
	}
	//y���˲�
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			v1 = v2 = v3 = 0;
			//���
			for (int k = 0; k < size; k++)
			{
				int p = j + size / 2 - k;
				if (p < 0) p *= -1;
				if (p > cols - 1) p = 2 * (cols - 1) - p;
				v1 += filter[k] * res[i * cols * 3 + p * 3 + 0];
				v2 += filter[k] * res[i * cols * 3 + p * 3 + 1];
				v3 += filter[k] * res[i * cols * 3 + p * 3 + 2];
			}
			res[i * cols * 3 + j * 3 + 0] = (uchar)int2smallint(v1);
			res[i * cols * 3 + j * 3 + 1] = (uchar)int2smallint(v2);
			res[i * cols * 3 + j * 3 + 2] = (uchar)int2smallint(v3);
		}
	}
	return res;
}