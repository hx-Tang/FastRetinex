#include <iostream>
#include <algorithm>
#include <ctime>
#include <omp.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace std;
using namespace cv;


//log2�����
void log2dic(double * Log2);
//matת����
void mat2arr(Mat, int *);
//����תmat
Mat arr2mat(double *,Mat);
//����ݹ�ϵ��
void Calc(int sigma, int &b0, int &b1, int &b2, int &b3);
//�ݹ��˹�˲�
void filterGaussian(int *BGR1, int cols, int rows, int b0, int b1, int b2, int b3);
//���߶�
void SSR(int *src,double *res, int sigma,int rows, int cols, double *Log2);
//��߶�
void MSR(double *res, double *res1, double *res2, int size);
//ɫ�ʻָ�
void MSRCR(int *src, double *res,int rows, int cols, double *);


/*
	�Ľ�˵��������fastRetinex.cpp
	1. �����������Ż�
	2. �ݹ��˹�˲��㷨
	3. openmp���м���
	4. ����ָ������ƶ���ѭ����


	Ч��������10��?


*/

void main() 
{
	//Ԥ��log2���㹩��� ����ʱ
	double * dic = new double[256*192];
	log2dic(dic);
	
	//��ʱ
	time_t start, stop;
	start = clock();

	//��RGBͼ
	Mat img = imread("test6.jpg");

	int size = img.rows * img.cols * 3;

	//���ͨ��������ά����
	int *BGR = new int[size];
	mat2arr(img, BGR);
	
	//SSR * 3
	double *res = new double[size];
	SSR(BGR, res, 80, img.rows, img.cols, dic);

	double *res1 = new double[size];
	//SSR(BGR, res1, 15, img.rows, img.cols, dic);

	double *res2 = new double[size];
	//SSR(BGR, res2, 200, img.rows, img.cols, dic);

	Mat img_ssr = arr2mat(res, img);

	//MSR
	//MSR(res, res1, res2, img.rows*img.cols);

	//Mat img_msr = arr2mat(res, img);

	//MSRCR
	MSRCR(BGR, res, img.rows, img.cols, dic);

	//Mat img_msrcr = arr2mat(res, img);

	stop = clock();

	cout << "Use Time:" << (double)(stop - start) / CLOCKS_PER_SEC << endl;

	//չʾ
	imshow("ԭͼ", img);
	imshow("���߶�Retinex", img_ssr);
	//imshow("��߶�Retinex", img_msr);
	//imshow("��߶�Retinex��ɫ�ʸ�ԭ", img_msrcr);

	waitKey(0);
}

//double����int��Сint��ת
inline int int2smallint(int x) { return (x >> 4); }
inline int double2int(double x) { return (int)(x * 16 + 0.5); }

//Ԥ��log2�����
void log2dic(double * dic)
{
	#pragma omp parallel for
	for (int i = 0; i < 256 * 192; i++) {
		dic[i] = log2(i);
	}
}

//��mat
void mat2arr(Mat img, int *BGR)
{
	//#pragma omp parallel for
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			for (int k = 0; k < 3; k++)
			{
				BGR[i * img.cols * 3 + j * 3 + k] = (int)img.at<Vec3b>(i, j)[k];
			}
		}
	}
}

//��װmat
Mat arr2mat(double *BGR, Mat src)
{
	double max = *max_element(BGR, (BGR + src.rows * src.cols * 3));
	double min = *min_element(BGR, (BGR + src.rows * src.cols * 3));
	cout << max << min << endl;
	Mat img;
	src.copyTo(img);

	//#pragma omp parallel for
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
void SSR(int *src, double *res, int sigma,int rows,int cols, double* Log2)
{
	//���ɵݹ�ϵ��
	int b0, b1, b2, b3;
	Calc(sigma, b0, b1, b2, b3);
	
	//�м����
	int *tmp = new int[(rows+6) * cols * 3];
	//���¸������з����˲�
	memcpy(tmp + 3 * cols * 3, src, rows * cols * 3 * sizeof(int));

	//�˲�
	filterGaussian(tmp,cols, rows, b0,b1,b2,b3);

	//������� ת��ʵ����
	for (int i = 0; i < rows * cols * 3; i++)
	{
		//res[i] = pow(2, (log2(src[i]) - log2(tmp[i])));
		//res[i] = pow(2, (Log2[src[i]] - Log2[tmp[i]]));
		res[i] = tmp[i];
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
void MSRCR(int *BGR, double *res , int rows, int cols, double * Log2)
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

/*
	�ݹ��˹�˲�
*/

//����ݹ�ϵ��b0��b1��b2��b3
void Calc(int sigma, int &b0, int &b1, int &b2, int &b3)
{
	//ȷ������˰뾶
	double Radius = sigma / 4;

	//��������㷨
	double Q, B, B1, B2, B3;
	Q = (double)(0.98711 * Radius - 0.96330);
	B = 1.57825 + 2.44413 * Q + 1.4281 * Q * Q + 0.422205 * Q * Q * Q;
	B1 = 2.44413 * Q + 2.85619 * Q * Q + 1.26661 * Q * Q * Q;
	B2 = -1.4281 * Q * Q - 1.26661 * Q * Q * Q;
	B3 = 0.422205 * Q * Q * Q;

	//ת��int
	b0 = double2int(1.0 - (B1 + B2 + B3) / B);
	b1 = double2int(B1 / B);
	b2 = double2int(B2 / B);
	b3 = double2int(B3 / B);
}



//��ά�ݹ��˹�˲� //���ö��߳�
void filterGaussian(int *tmp, int cols, int rows, int b0, int b1, int b2, int b3)
{

	//X���˲�

	//������
//#pragma omp parallel for
	for (int i = 0; i < rows; i++)
	{
		// �������Ҷ�����
		int *prL = (tmp + 3 * cols * 3) + i * cols * 3;

		// ��Ե���ظ�����
		int B1 = prL[0], B2 = prL[0], B3 = prL[0];
		int G1 = prL[1], G2 = prL[1], G3 = prL[1];
		int R1 = prL[2], R2 = prL[2], R3 = prL[2];
		for (int j = 0; j < cols; j++, prL += 3)
		{
			// ˳�����
			//prL[0] = int2smallint(prL[0] * b0 + B1 * b1 + B2 * b2 + B3 * b3);
			//prL[1] = int2smallint(prL[1] * b0 + G1 * b1 + G2 * b2 + G3 * b3);
			//prL[2] = int2smallint(prL[2] * b0 + R1 * b1 + R2 * b2 + R3 * b3);
			//prL[0] = prL[0] * b0 + B1 * b1 + B2 * b2 + B3 * b3;
			//prL[1] = prL[1] * b0 + G1 * b1 + G2 * b2 + G3 * b3;
			//prL[2] = prL[2] * b0 + R1 * b1 + R2 * b2 + R3 * b3;
			cout << prL[0] <<"|"<< prL[1] <<"|"<< prL[2] << endl;
			// ǰ����λ
			B3 = B2, B2 = B1, B1 = prL[0];
			G3 = G2, G2 = G1, G1 = prL[1];
			R3 = R2, R2 = R1, R1 = prL[2];
		}

		// �������� 
		int *prR = (tmp + 3 * cols * 3) + (i + 1) * cols * 3 - 3;

		// ��Ե���ظ�����
		int B1R = prR[0], B2R = prR[0], B3R = prR[0];
		int G1R = prR[1], G2R = prR[1], G3R = prR[1];
		int R1R = prR[2], R2R = prR[2], R3R = prR[2];
		for (int j = 0; j < cols; j++, prR -= 3)
		{
			// �������
			prR[0] = int2smallint(prR[0] * b0 + B1R * b1 + B2R * b2 + B3R * b3);
			prR[1] = int2smallint(prR[1] * b0 + G1R * b1 + G2R * b2 + G3R * b3);
			prR[2] = int2smallint(prR[2] * b0 + R1R * b1 + R2R * b2 + R3R * b3);

			// ������λ
			B3R = B2R, B2R = B1R, B1R = prR[0];
			G3R = G2R, G2R = G1R, G1R = prR[1];
			R3R = R2R, R2R = R1R, R1R = prR[2];
		}
	}

	//Y���˲�

	//����ͷ����BGR����
	memcpy(tmp + 0 * cols * 3, tmp + 3 * cols * 3, cols * 3 * sizeof(int));
	memcpy(tmp + 1 * cols * 3, tmp + 3 * cols * 3, cols * 3 * sizeof(int));
	memcpy(tmp + 2 * cols * 3, tmp + 3 * cols * 3, cols * 3 * sizeof(int));
	//����β����BGR����
	memcpy(tmp + (rows + 3) * cols * 3, tmp + (rows + 2) * cols * 3, cols * 3 * sizeof(int));
	memcpy(tmp + (rows + 4) * cols * 3, tmp + (rows + 2) * cols * 3, cols * 3 * sizeof(int));
	memcpy(tmp + (rows + 5) * cols * 3, tmp + (rows + 2) * cols * 3, cols * 3 * sizeof(int));

	//�����µ���
	for (int i = 0; i < rows; i++)
	{
		//�趨��ʼֵ
		int *pr3 = tmp + (i + 0) * cols * 3;
		int *pr2 = tmp + (i + 1) * cols * 3;
		int *pr1 = tmp + (i + 2) * cols * 3;
		int *pr0 = tmp + (i + 3) * cols * 3;
		//������������һ�е�����
		for (int j = 0; j < cols; j++, pr0 += 3, pr1 += 3, pr2 += 3, pr3 += 3)
		{
			// ˳�����
			pr0[0] = int2smallint(pr0[0] * b0 + pr1[0] * b1 + pr2[0] * b2 + pr3[0] * b3);
			pr0[1] = int2smallint(pr0[1] * b0 + pr1[1] * b1 + pr2[1] * b2 + pr3[1] * b3);
			pr0[2] = int2smallint(pr0[2] * b0 + pr1[2] * b1 + pr2[2] * b2 + pr3[2] * b3);
		}
	}

	//�����ϵ���
	for (int i = rows - 1; i >= 0; i--)
	{
		// �趨��ʼֵ
		int *pr3 = tmp + (i + 3) * cols * 3;
		int *pr2 = tmp + (i + 2) * cols * 3;
		int *pr1 = tmp + (i + 1) * cols * 3;
		int *pr0 = tmp + (i + 0) * cols * 3;
		//������������һ�е�����
		for (int j = 0; j < cols; j++, pr0 += 3, pr1 += 3, pr2 += 3, pr3 += 3)
		{
			// �������
			pr0[0] = int2smallint(pr0[0] * b0 + pr1[0] * b1 + pr2[0] * b2 + pr3[0] * b3);
			pr0[1] = int2smallint(pr0[1] * b0 + pr1[1] * b1 + pr2[1] * b2 + pr3[1] * b3);
			pr0[2] = int2smallint(pr0[2] * b0 + pr1[2] * b1 + pr2[2] * b2 + pr3[2] * b3);
			//cout << pr0[0] << pr0[1] << pr0[2] << endl;
		}
	}
}



//��ά�ݹ��˹�˲� //���ö��߳�
void filterGaussian__(double *tmp, int cols, int rows, double b0, double b1, double b2, double b3)
{

	//X���˲�

	//������
	#pragma omp parallel for
	for (int i = 0; i < rows; i++)
	{
		// �������Ҷ�����
		double *prL = (tmp + 3 * cols * 3) + i * cols * 3;

		// ��Ե���ظ�����
		double B1 = prL[0], B2 = prL[0], B3 = prL[0];
		double G1 = prL[1], G2 = prL[1], G3 = prL[1];
		double R1 = prL[2], R2 = prL[2], R3 = prL[2];
		for (int j = 0; j < cols; j++, prL += 3)
		{
			// ˳�����
			prL[0] = prL[0] * b0 + B1 * b1 + B2 * b2 + B3 * b3;
			prL[1] = prL[1] * b0 + G1 * b1 + G2 * b2 + G3 * b3;
			prL[2] = prL[2] * b0 + R1 * b1 + R2 * b2 + R3 * b3;

			// ǰ����λ
			B3 = B2, B2 = B1, B1 = prL[0];
			G3 = G2, G2 = G1, G1 = prL[1];
			R3 = R2, R2 = R1, R1 = prL[2];
		}
		// �������� 
		double *prR = (tmp + 3 * cols * 3) + (i + 1) * cols * 3 - 3;

		// ��Ե���ظ�����
		double B1R = prR[0], B2R = prR[0], B3R = prR[0];
		double G1R = prR[1], G2R = prR[1], G3R = prR[1];
		double R1R = prR[2], R2R = prR[2], R3R = prR[2];
		for (int j = 0; j < cols; j++, prR -= 3)
		{
			// �������
			prR[0] = prR[0] * b0 + B1R * b1 + B2R * b2 + B3R * b3;
			prR[1] = prR[1] * b0 + G1R * b1 + G2R * b2 + G3R * b3;
			prR[2] = prR[2] * b0 + R1R * b1 + R2R * b2 + R3R * b3;

			// ������λ
			B3R = B2R, B2R = B1R, B1R = prR[0];
			G3R = G2R, G2R = G1R, G1R = prR[1];
			R3R = R2R, R2R = R1R, R1R = prR[2];
		}
	}

	//Y���˲�

	//����ͷ����BGR����
	memcpy(tmp + 0 * cols * 3, tmp + 3 * cols * 3, cols * 3 * sizeof(double));
	memcpy(tmp + 1 * cols * 3, tmp + 3 * cols * 3, cols * 3 * sizeof(double));
	memcpy(tmp + 2 * cols * 3, tmp + 3 * cols * 3, cols * 3 * sizeof(double));
	//����β����BGR����
	memcpy(tmp + (rows + 3) * cols * 3, tmp + (rows + 2) * cols * 3, cols * 3 * sizeof(double));
	memcpy(tmp + (rows + 4) * cols * 3, tmp + (rows + 2) * cols * 3, cols * 3 * sizeof(double));
	memcpy(tmp + (rows + 5) * cols * 3, tmp + (rows + 2) * cols * 3, cols * 3 * sizeof(double));

	//�����µ���
	//#pragma omp parallel for ordered
	for (int i = 0; i < rows; i++)
	{
		//�趨��ʼֵ
		double *pr3 = tmp + (i + 0) * cols * 3;
		double *pr2 = tmp + (i + 1) * cols * 3;
		double *pr1 = tmp + (i + 2) * cols * 3;
		double *pr0 = tmp + (i + 3) * cols * 3;
		//������������һ�е�����
		for (int j = 0; j < cols; j++, pr0 += 3, pr1 += 3, pr2 += 3, pr3 += 3)
		{
			// ˳�����
			pr0[0] = pr0[0] * b0 + pr1[0] * b1 + pr2[0] * b2 + pr3[0] * b3;
			pr0[1] = pr0[1] * b0 + pr1[1] * b1 + pr2[1] * b2 + pr3[1] * b3;
			pr0[2] = pr0[2] * b0 + pr1[2] * b1 + pr2[2] * b2 + pr3[2] * b3;
		}
	}

	//�����ϵ���
	//#pragma omp parallel for ordered
	for (int i = rows - 1; i >= 0; i--)
	{
		// �趨��ʼֵ
		double *pr3 = tmp + (i + 3) * cols * 3;
		double *pr2 = tmp + (i + 2) * cols * 3;
		double *pr1 = tmp + (i + 1) * cols * 3;
		double *pr0 = tmp + (i + 0) * cols * 3;
		//������������һ�е�����
		for (int j = 0; j < cols; j++, pr0 += 3, pr1 += 3, pr2 += 3, pr3 += 3)
		{
			// �������
			pr0[0] = pr0[0] * b0 + pr1[0] * b1 + pr2[0] * b2 + pr3[0] * b3;
			pr0[1] = pr0[1] * b0 + pr1[1] * b1 + pr2[1] * b2 + pr3[1] * b3;
			pr0[2] = pr0[2] * b0 + pr1[2] * b1 + pr2[2] * b2 + pr3[2] * b3;
		}
	}
}

//����ݹ�ϵ��b0��b1��b2��b3
void Calc__(int sigma, double &b0, double &b1, double &b2, double &b3)
{
	//ȷ������˰뾶 
	//���γ��Ժ� 1/4��sigma Ч���ȽϺ� 
	float Radius = sigma / 4;

	//��������㷨
	float Q, B, B1, B2, B3;
	Q = (double)(0.98711 * Radius - 0.96330);
	B = 1.57825 + 2.44413 * Q + 1.4281 * Q * Q + 0.422205 * Q * Q * Q;
	B1 = 2.44413 * Q + 2.85619 * Q * Q + 1.26661 * Q * Q * Q;
	B2 = -1.4281 * Q * Q - 1.26661 * Q * Q * Q;
	B3 = 0.422205 * Q * Q * Q;

	//��һ��
	b0 = 1.0 - (B1 + B2 + B3) / B;
	b1 = B1 / B;
	b2 = B2 / B;
	b3 = B3 / B;
}

