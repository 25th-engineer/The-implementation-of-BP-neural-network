#include"bp.h"

Back_propagation::Back_propagation()
{
	srand( (unsigned)time( NULL ) );
	for( int i = 0; i < HN; i ++ )
	{
		for( int j = 0; j < INnum; j ++ )
		{
			W[i][j] = double( rand() % 100 ) / 100; //初始化输入层到隐层的权值	
		}
	}
	for( int ii = 0; ii < ONnum; ii ++ )
	{
		for( int jj = 0; jj < HN; jj ++ )
		{
			V[ii][jj] = double( rand() % 100 ) / 100;  //初始化隐层到输出层的权值
		}
	}
}

void Back_propagation::input_p(int m)
{
   	for( int i = 0; i < INnum; i ++ )
	{
		P[i] = Study_Data[m].input[i];
	}
}

void Back_propagation::input_t(int m)
{
	for( int k = 0; k < ONnum; k ++ )
	{
		T[k] = Study_Data[m].teach[k];
	}
}

void Back_propagation::H_I_O()
{
	double net;
	int i,j;
	for( j = 0; j < HN; j ++ )
	{
		net = 0;
		for( i = 0; i < INnum; i ++ )
		{
			net += W[j][i] * P[i];//求隐层内积	
		}
		HI[j] = net;// - Thread_Hiden[j];//求隐层输入
		hidenLayerOutput[j] = 1.0 / ( 1.0 + exp(-HI[j]) );//求隐层输出 
	}
}

void Back_propagation::O_I_O()
{
	double net;
	int k,j;
	for( k = 0; k < ONnum; k ++ )
	{
		net = 0;
		for( j = 0; j < HN; j ++ )
		{
			net += V[k][j] * hidenLayerOutput[j];//求输出层内积
		}
		OI[k] = net; //求输出层输入
		OO[k] = 1.0 / ( 1.0 + exp(-OI[k]) );//求输出层输出
	}
}

void Back_propagation::Err_Output_Hidden( int m )
{
	double abs_err[ONnum];//样本误差
	double sqr_err = 0;//临时保存误差平方
	
	for( int k = 0; k < ONnum; k ++ )
	{
		abs_err[k] = T[k] - OO[k];	//求第m个样本下的第k个神经元的绝对误差

		sqr_err += (abs_err[k]) * (abs_err[k]);//求第m个样本下输出层的平方误差

		d_err[k] = abs_err[k] * OO[k] * (1.0-OO[k]);//d_err[k]输出层各神经元的一般化误差
	}
	err_m[m] = sqr_err / 2;//第m个样本下输出层的平方误差/2=第m个样本的均方误差,据ppt1.5-3

}

void Back_propagation::Err_Hidden_Input()
{
  double sigma;
  for( int j = 0; j < HN; j ++ ) 
  {
	 sigma = 0.0;
     for( int k = 0; k < ONnum; k ++ ) 
	 {
        sigma += d_err[k] * V[k][j];  
	 }
	 e_err[j] = sigma * hidenLayerOutput[j] * ( 1 - hidenLayerOutput[j] );//隐层各神经元的一般化误差
  }
}

void Back_propagation::Adjust_O_H( int m,int n )
{
	if( n <= 1 )
	{
		for( int k = 0; k < ONnum; k ++ )
		{
			for( int j = 0; j < HN; j ++ )
			{
				V[k][j] = V[k][j] + studyRate * d_err[k] * hidenLayerOutput[j];//输出层至隐层的权值调整
			}
		}
	}
	else if( n > 1 )
	{
		for( int k = 0; k < ONnum; k ++ )
        {
			for( int j = 0; j < HN; j ++ )
			{
				V[k][j] = V[k][j] + studyRate * d_err[k] * hidenLayerOutput[j] + b * ( V[k][j] - OLD_V[k][j] );//输出层至隐层的权值调整
			}
		}
	}
}

void Back_propagation::Adjust_H_I( int m,int n )
{
	if( n <= 1 )
	{
		for( int j = 0; j < HN; j ++ )
		{
			for ( int i = 0; i < INnum; i ++ ) 
			{
				W[j][i] = W[j][i] + studyRate * e_err[j] * P[i];//隐层至输入层的权值调整
			}
		}

	}
	else if( n > 1 )
	{
		for( int j = 0; j < HN; j ++ )
		{
			for( int i = 0; i < INnum; i ++ ) 
			{
				W[j][i] += studyRate * e_err[j] * P[i] + b * ( W[j][i] - OLD_W[j][i] );//隐层至输入层的权值调整
			}
	    }
	}
}

void Back_propagation::saveWV()
{
	for( int i = 0; i < HN; i ++ )
	{
		for( int j = 0; j < INnum; j ++ )
		{
			OLD_W[i][j] = W[i][j];
		}
	}

	for( int ii = 0; ii < ONnum; ii ++ )
	{
		for( int jj = 0; jj < HN; jj ++ )
		{
			OLD_V[ii][jj] = V[ii][jj];
		}
	}
}
