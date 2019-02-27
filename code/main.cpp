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

//保存数据
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
	  total_err += bp.err_m[m];//每个样本的均方误差加起来就成了全局误差
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

	cout << "请输入学习效率: studyRate = ";
	cin >> bp.studyRate;

	cout << "\n请输入步长: b= ";
	cin >> bp.b;

	study = 0;
	double Pre_error ; //预定误差
	cout << "\n请输入预定误差: Pre_error = ";
	cin >> Pre_error;

	int Pre_times;
	cout << "\n请输入预定最大学习次数:Pre_times=";
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
			cout << "训练失败！" << endl;
			break;
		}

		for ( int m = 0; m < SampleCount; m ++ ) 
		{
			bp.input_p(m); //输入第m个学习样本 (2)
			bp.input_t(m);//输入第m个样本的教师信号 (3)
			bp.H_I_O(); //第m个学习样本隐层各单元输入、输出值 (4)
			bp.O_I_O();
			bp.Err_Output_Hidden(m); //第m个学习样本输出层至隐层一般化误差 (6) 
			bp.Err_Hidden_Input(); //第m个学习样本隐层至输入层一般化误差 (7)
			bp.Adjust_O_H(m,study);
			bp.Adjust_H_I(m,study);
			if( m == 0 )
			{
				cout << bp.V[0][0] << " " << bp.V[0][1] << endl;
			}
		}//全部样本训练完毕
		sum_err = Err_Sum(bp); //全部样本全局误差计算 (10)
		bp.saveWV();
	}while( sum_err > Pre_error ); 

	if( ( study <= Pre_times ) & ( sum_err < Pre_error ) )
	{
		cout << "训练结束！" << endl;
		cout << "你已经学习了 " << study << "次" << endl;
	}
    double net;
	int k, j;
	while(1)
	{
		printf( "请输入蠓虫的触角及翅膀的长度:" );
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
				net += bp.V[k][j] * bp.hidenLayerOutput[j];//求输出层内积
			}
			bp.OI[k] = net; //求输出层输入
			bp.OO[k] = 1.0 / ( 1.0 + exp(-bp.OI[k]) );//求输出层输出
		}
		if( bp.OO[0] > 0.5 )
		{
			printf( "该蠓虫是af!\n" );
		}
		else if( bp.OO[0] >= 0 )
		{
			printf( "该蠓虫是apf!\n" );
		}
    }
	return 0;
}

