#ifndef BP_H
#define BP_H

#include "necessary.h"


class Back_propagation
{

public:
	Back_propagation();
    double W[HN][INnum]; //输入层至隐层权值
	double V[ONnum][HN]; //隐层至输出层权值
	double P[INnum]; //单个样本输入数据
    double T[ONnum]; //单个样本期望输出值

    double OLD_W[HN][INnum];  //保存HN-IN旧权！
    double OLD_V[ONnum][HN];  //保存ON-HN旧权！
    double HI[HN]; //隐层的输入
    double OI[ONnum]; //输出层的输入
    double hidenLayerOutput[HN]; //隐层的输出
	double OO[ONnum]; //输出层的输出
	double err_m[SampleCount]; //第m个样本的总误差
	double studyRate;//学习效率效率
	double b;//步长
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