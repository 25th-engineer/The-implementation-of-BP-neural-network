#include "bp.h"

void saveWV( Back_propagation bp )
{
	for( int i = 0; i < HN; i ++ )
	{
		for( int j = 0; j < INnum; j ++ )
		{
			bp.OLD_W[i][j] = bp.W[i][j];
		}
	}

	for( int ii = 0; ii < ONnum; ii ++ )
	{
		for( int jj = 0; jj < HN; jj ++ )
		{
			bp.OLD_V[ii][jj] = bp.V[ii][jj];
		}
	}
}

//��������
void savequan( Back_propagation bp )
{
	ofstream outW( "w.txt" );
	ofstream outV( "v.txt" );
	
	for( int i = 0; i < HN; i ++ )
	{
		for( int j = 0; j < INnum; j ++ )
		{
			outW << bp.W[i][j] << "   ";		
		}
		outW << "\n";
	}

	for( int ii = 0; ii < ONnum; ii ++ )
	{
		for( int jj = 0; jj < HN; jj ++ )
		{
			outV << bp.V[ii][jj] << "   ";
		}
		outV << "\n";
	}


	outW.close();
	outV.close();

}

double Err_Sum( Back_propagation bp )
{
	double total_err = 0;
	for ( int m = 0; m < SampleCount; m ++ ) 
	{
	  total_err += bp.err_m[m];//ÿ�������ľ������������ͳ���ȫ�����
	}
	return total_err;
}

int main()
{
	double sum_err;
	int study;
	int m;
    double check_in, check_out;
    ifstream Train_in( "trainin.txt", ios::in );
	ifstream Train_out( "trainout.txt", ios::in );

	if( ( Train_in.fail() ) || ( Train_out.fail() ) )
	{
	    //printf( "Error input file!\n" );
		cerr << "Error input file!" << endl;
		exit(0);
	}

	Back_propagation bp;

	cout << "������ѧϰЧ��: studyRate = ";
	cin >> bp.studyRate;

	cout << "\n�����벽��: b= ";
	cin >> bp.b;

	study = 0;
	double Pre_error ; //Ԥ�����
	cout << "\n������Ԥ�����: Pre_error = ";
	cin >> Pre_error;

	int Pre_times;
	cout << "\n������Ԥ�����ѧϰ����:Pre_times=";
	cin >> Pre_times;

	for( m = 0; m < SampleCount; m ++ )
	{
		for( int i = 0; i < INnum; i ++ )
		{
			Train_in >> bp.Study_Data[m].input[i];
		}
	}
 
	cout << endl;
	for( m = 0; m < SampleCount; m ++ )
	{
		for( int k = 0; k < ONnum; k ++ )
		{
	       Train_out >> bp.Study_Data[m].teach[k];
		}
	}
	
	cout << endl;

	do
	{
		++ study;
		if( study > Pre_times )
		{
			cout << "ѵ��ʧ�ܣ�" << endl;
			break;
		}

		for ( int m = 0; m < SampleCount; m ++ ) 
		{
			bp.input_p(m); //�����m��ѧϰ���� (2)
			bp.input_t(m);//�����m�������Ľ�ʦ�ź� (3)
			bp.H_I_O(); //��m��ѧϰ�����������Ԫ���롢���ֵ (4)
			bp.O_I_O();
			bp.Err_Output_Hidden(m); //��m��ѧϰ���������������һ�㻯��� (6) 
			bp.Err_Hidden_Input(); //��m��ѧϰ���������������һ�㻯��� (7)
			bp.Adjust_O_H(m,study);
			bp.Adjust_H_I(m,study);
			if( m == 0 )
			{
				cout << bp.V[0][0] << " " << bp.V[0][1] << endl;
			}
		}//ȫ������ѵ�����
		sum_err = Err_Sum(bp); //ȫ������ȫ�������� (10)
		bp.saveWV();
	}while( sum_err > Pre_error ); 

	if( ( study <= Pre_times ) & ( sum_err < Pre_error ) )
	{
		cout << "ѵ��������" << endl;
		cout << "���Ѿ�ѧϰ�� " << study << "��" << endl;
	}
    double net;
	int k, j;
	while(1)
	{
		printf( "��������Ĵ��Ǽ����ĳ���:" );
		cin >> check_in;
		cin >> check_out;
		bp.P[0] = check_in;
		bp.P[1] = check_out;
		bp.H_I_O();
		for ( k = 0; k < ONnum; k ++ )
		{
			net = 0;
			for( j = 0; j < HN; j ++ )
			{
				net += bp.V[k][j] * bp.hidenLayerOutput[j];//��������ڻ�
			}
			bp.OI[k] = net; //�����������
			bp.OO[k] = 1.0 / ( 1.0 + exp(-bp.OI[k]) );//����������
		}
		if( bp.OO[0] > 0.5 )
		{
			printf( "������af!\n" );
		}
		else if( bp.OO[0] >= 0 )
		{
			printf( "������apf!\n" );
		}
    }
	return 0;
}

