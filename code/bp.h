#ifndef BP_H
#define BP_H

#include "necessary.h"


class Back_propagation
{

public:
	Back_propagation();
    double W[HN][INnum]; //�����������Ȩֵ
	double V[ONnum][HN]; //�����������Ȩֵ
	double P[INnum]; //����������������
    double T[ONnum]; //���������������ֵ

    double OLD_W[HN][INnum];  //����HN-IN��Ȩ��
    double OLD_V[ONnum][HN];  //����ON-HN��Ȩ��
    double HI[HN]; //���������
    double OI[ONnum]; //����������
    double hidenLayerOutput[HN]; //��������
	double OO[ONnum]; //���������
	double err_m[SampleCount]; //��m�������������
	double studyRate;//ѧϰЧ��Ч��
	double b;//����
	double e_err[HN];
	double d_err[ONnum];
	void input_p(int m);
	void input_t(int m);
    void H_I_O();
	void O_I_O();
	void Err_Output_Hidden(int m);
	void Err_Hidden_Input();
	void Adjust_O_H(int m,int n);
	void Adjust_H_I(int m,int n);
    void saveWV();
	struct
	{
		double input[INnum];
	    double teach[ONnum];
	}Study_Data[SampleCount];
private:
};

#endif