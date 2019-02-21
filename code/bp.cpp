#include"bp.h"

Back_propagation::Back_propagation()
{
	srand( (unsigned)time( NULL ) );
	for( int i = 0; i < HN; i ++ )
	{
		for( int j = 0; j < INnum; j ++ )
		{
			W[i][j] = double( rand() % 100 ) / 100; //��ʼ������㵽�����Ȩֵ	
		}
	}
	for( int ii = 0; ii < ONnum; ii ++ )
	{
		for( int jj = 0; jj < HN; jj ++ )
		{
			V[ii][jj] = double( rand() % 100 ) / 100;  //��ʼ�����㵽������Ȩֵ
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
			net += W[j][i] * P[i];//�������ڻ�	
		}
		HI[j] = net;// - Thread_Hiden[j];//����������
		hidenLayerOutput[j] = 1.0 / ( 1.0 + exp(-HI[j]) );//��������� 
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
			net += V[k][j] * hidenLayerOutput[j];//��������ڻ�
		}
		OI[k] = net; //�����������
		OO[k] = 1.0 / ( 1.0 + exp(-OI[k]) );//����������
	}
}

void Back_propagation::Err_Output_Hidden( int m )
{
	double abs_err[ONnum];//�������
	double sqr_err = 0;//��ʱ�������ƽ��
	
	for( int k = 0; k < ONnum; k ++ )
	{
		abs_err[k] = T[k] - OO[k];	//���m�������µĵ�k����Ԫ�ľ������

		sqr_err += (abs_err[k]) * (abs_err[k]);//���m��������������ƽ�����

		d_err[k] = abs_err[k] * OO[k] * (1.0-OO[k]);//d_err[k]��������Ԫ��һ�㻯���
	}
	err_m[m] = sqr_err / 2;//��m��������������ƽ�����/2=��m�������ľ������,��ppt1.5-3

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
	 e_err[j] = sigma * hidenLayerOutput[j] * ( 1 - hidenLayerOutput[j] );//�������Ԫ��һ�㻯���
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
				V[k][j] = V[k][j] + studyRate * d_err[k] * hidenLayerOutput[j];//������������Ȩֵ����
			}
		}
	}
	else if( n > 1 )
	{
		for( int k = 0; k < ONnum; k ++ )
        {
			for( int j = 0; j < HN; j ++ )
			{
				V[k][j] = V[k][j] + studyRate * d_err[k] * hidenLayerOutput[j] + b * ( V[k][j] - OLD_V[k][j] );//������������Ȩֵ����
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
				W[j][i] = W[j][i] + studyRate * e_err[j] * P[i];//������������Ȩֵ����
			}
		}

	}
	else if( n > 1 )
	{
		for( int j = 0; j < HN; j ++ )
		{
			for( int i = 0; i < INnum; i ++ ) 
			{
				W[j][i] += studyRate * e_err[j] * P[i] + b * ( W[j][i] - OLD_W[j][i] );//������������Ȩֵ����
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
