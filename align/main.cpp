#include<iostream>
#include<pmmintrin.h> //SSE3
#include<windows.h>


using namespace std;


float** normal_gaosi(float **A, int n){
	//串行算法主体部分
for (int k=0;k<n;k++){

      //将第k行第k个元素全部变为1
      for(int j=k+1;j<n;j++)
           {
			A[k][j] = A[k][j]/A[k][k];
	       }
      A[k][k]=1.0;


      //对k+1到第n-1行进行高斯消元
      for(int i=k+1;i<n;i++){
           for(int j=k+1;j<n;j++)
                 A[i][j] = A[i][j]-A[i][k]*A[k][j];
       A[i][k]=0;
           }
         }

     return A;
}

void display(float **A, int n){

        for(int i = 0;i<n; i++){
	          for(int j=0; j<n; j++)
		           {cout<<A[i][j]<<' ';}
                          cout<<endl;}

}




float** SSE_gaosi_part1(float **A, int n){
    //SSE算法——part1
    __m128 t1, t2, t3;

    for (int k=0;k<n;k++){

      //将第k行第k个元素全部变为1
      float tmp[4] = {A[k][k], A[k][k], A[k][k], A[k][k]}; //创建一个浮点数向量A[k][k]
      t1 = _mm_loadu_ps(tmp);  //将向量float装载进t1中 {*不对齐形式}

      int j = n-4;
      for (j = n-4; j >= k ; j = j-4)

        {
            t2 = _mm_loadu_ps(A[k] + j);
            t3 = _mm_div_ps(t2, t1);
            _mm_storeu_ps(A[k] + j, t3);

        }

       if((j+4)!=k){
                for(int i=k; (i<n)&&(i<(j+4)); i++){
                    A[k][i] = A[k][i]/A[k][k];
                }
        }

      //对k+1到第n-1行进行高斯消元

     for(int i=k+1;i<n;i++){
           for(int j=k+1;j<n;j++)
                 A[i][j] = A[i][j]-A[i][k]*A[k][j];
       A[i][k]=0;
           }
         }
     /*for(int i=0;i<n;i++){
     for(int j=0;j<n;j++){
           cout<<A[i][j]<<' ';}
           cout<<endl;}
    display(A,n);*/

     return A;
}

float** SSE_gaosi_part2(float **A, int n){

     __m128 t1, t2, t3, t4;
    //SSE算法——part2
    for (int k=0;k<n;k++){

      //将第k行第k个元素全部变为1-串行
      for(int j=k+1;j<n;j++)
           {
			A[k][j] = A[k][j]/A[k][k];

	        }
      A[k][k]=1.0;

      //对k+1到第n-1行进行高斯消元
      //并行SSE-part2

      for(int i=k+1;i<n;i++){

           float tmp[4] = {A[i][k], A[i][k], A[i][k], A[i][k]};
           t1 = _mm_loadu_ps(tmp);

           int j;
           for(j=n-4;j>=k;j=j-4){

                 //A[i][j] = A[i][j]-A[i][k]*A[k][j];
                 t2 = _mm_loadu_ps(A[i] + j);
                 t3 = _mm_loadu_ps(A[k] + j);
                 t4 = _mm_sub_ps(t2, _mm_mul_ps(t1, t3));
                 _mm_storeu_ps(A[i]+j, t4);

                                  }
           if((j+4)!=k)
      {
          for(int s=k; s<(j+4);s++)
              A[i][s] = A[i][s]-A[i][k]*A[k][s];

      }
           }
     }
     return A;

}


float** SSE_gaosi_all(float **A, int n){
    //SSE算法——all
    __m128 t1, t2, t3, t4;


    for (int k=0;k<n;k++){
      float tmp[4] = {A[k][k], A[k][k], A[k][k], A[k][k]}; //创建一个浮点数向量A[k][k]
      t1 = _mm_loadu_ps(tmp);  //将向量float装载进t1中 {*不对齐形式}

      int j = n-4;
      for(j=n-4;j>=k;j = j-4)
           {
			//将A[k][j] = A[k][j]/A[k][k]并行化;
            t2 = _mm_loadu_ps(A[k]+j);   //将第k行的j~j+4列元素装载进来
            t3 = _mm_div_ps(t2, t1); //向量除
            _mm_storeu_ps(A[k]+j, t3);   //将向量结果装载回去

            }
      //处理不能被4整除的数
        if((j+4)!=k){
                for(int i=k; (i<n)&&(i<(j+4)); i++){
                    A[k][i] = A[k][i]/A[k][k];
                }
        }


      //对k+1到第n-1行进行高斯消元
      for(int i=k+1;i<n;i++){

           float tmp[4] = {A[i][k], A[i][k], A[i][k], A[i][k]};
           t1 = _mm_loadu_ps(tmp);

           int j = n-4;
           for(j=n-4;j>=k;j=j-4){
                 //A[i][j] = A[i][j]-A[i][k]*A[k][j];
                 t2 = _mm_loadu_ps(A[i] + j);
                 t3 = _mm_loadu_ps(A[k] + j);
                 t4 = _mm_sub_ps(t2, _mm_mul_ps(t1, t3));
                 _mm_storeu_ps(A[i]+j, t4);}

           if((j+4)!=k)
      {
          for(int s=k; s<(j+4);s++)
              A[i][s] = A[i][s]-A[i][k]*A[k][s];

      }
           }
    }
     return A;

}

float** deepcopy(float** A, float** S, int n){
    A = new float*[n];
    for(int i=0;i<n;i++){
        A[i] = new float[n];
    }
    for(int i=0;i<n;i++)
         for(int j=0;j<n;j++)
    {
        A[i][j]=S[i][j];

    }
    return A;
}

float ** aline(float **A, int n){

     __m128 t1, t2, t3, t4;


    for (int k=0;k<n;k++){
      float tmp[4] = {A[k][k], A[k][k], A[k][k], A[k][k]}; //创建一个浮点数向量A[k][k]
      t1 = _mm_load_ps(tmp);  //将向量float装载进t1中 {*不对齐形式}

      int j = k+1;
      for(;j+4<n;j=j+4)
        {
           while(j%4!=0&&j<n){
               A[k][j]=A[k][j]/A[k][k];
               j++;

           }
           if(j+4<n){
            t2 = _mm_load_ps(A[k]+j);   //将第k行的j~j+4列元素装载进来
            t3 = _mm_div_ps(t2, t1); //向量除
            _mm_store_ps(A[k]+j, t3);
            }

          for(;j<n;j++)
           {
            A[k][j]=A[k][j]/A[k][k];
             }
            A[k][k]=1.0;

         }


      //对k+1到第n-1行进行高斯消元
      for(int i=k+1;i<n;i++){

           float tmp[4] = {A[i][k], A[i][k], A[i][k], A[i][k]};
           t1 = _mm_load_ps(tmp);

           int j = k+1;
      for(;j+4<n;j=j+4)
        {
           while(j%4!=0&&j<n){
               A[i][j] = A[i][j]-A[i][k]*A[k][j];
               j++;

           }
           if(j+4<n){
                 t2 = _mm_load_ps(A[i] + j);
                 t3 = _mm_load_ps(A[k] + j);
                 t4 = _mm_sub_ps(t2, _mm_mul_ps(t1, t3));
                 _mm_store_ps(A[i]+j, t4);
            }

          for(;j<n;j++)
           {
            A[i][j] = A[i][j]-A[i][k]*A[k][j];
             }

         }

           /*
           int j = n-4;
           for(j=n-4;j>=k;j=j-4){
                 //A[i][j] = A[i][j]-A[i][k]*A[k][j];
                 t2 = _mm_loadu_ps(A[i] + j);
                 t3 = _mm_loadu_ps(A[k] + j);
                 t4 = _mm_sub_ps(t2, _mm_mul_ps(t1, t3));
                 _mm_storeu_ps(A[i]+j, t4);}

           if((j+4)!=k)
      {
          for(int s=k; s<(j+4);s++)
              A[i][s] = A[i][s]-A[i][k]*A[k][s];

      }

           */
    }
    }
     return A;





}




int main(){

for(int n=10;n<=1280;n=n*2){
        cout<<n<<' ';
//生成矩阵
float** A = new float* [n];
for(int i=0;i<n;i++)
    A[i]= new float[n];


for(int i=0;i<n;i++)
      { for(int j=0;j<i;j++)
                 A[i][j]=0;
                 A[i][i]=1.0;
        for(int j=i+1;j<n;j++)
                  A[i][j]=rand();
       }
for(int k=0;k<n;k++)
       for(int i=k+1;i<n;i++)
              for(int j=0;j<n;j++)
                        A[i][ j]+=A[k][j];



float **A2=nullptr;
A2 = deepcopy(A2, A, n);
float **A3=nullptr;
A3 = deepcopy(A3, A, n);
float **A4=nullptr;
A4 = deepcopy(A4, A, n);



long long head, tail , freq ;
QueryPerformanceFrequency((LARGE_INTEGER *)&freq );



QueryPerformanceCounter((LARGE_INTEGER *)&head);
for(int i=0;i<10;i++){
A = normal_gaosi(A,n);
}
QueryPerformanceCounter((LARGE_INTEGER *)&tail );
cout<<(tail - head)*1000.0/freq<<' ';



/*
QueryPerformanceCounter((LARGE_INTEGER *)&head);
for(int i=0;i<1;i++){
A2 = SSE_gaosi_part1(A2,n);
}
QueryPerformanceCounter((LARGE_INTEGER *)&tail );
cout<<"普通高斯SSE（part1）Time: "<<(tail - head)*1000.0/freq<<"ms"<<endl;


QueryPerformanceCounter((LARGE_INTEGER *)&head);
for(int i=0;i<1;i++){
A3 = SSE_gaosi_part2(A3,n);
}
QueryPerformanceCounter((LARGE_INTEGER *)&tail );
cout<<"普通高斯SSE（part2）Time: "<<(tail - head)*1000.0/freq<<"ms"<<endl;

*///不对齐
QueryPerformanceCounter((LARGE_INTEGER *)&head);
for(int i=0;i<10;i++){
A3 = SSE_gaosi_all(A4,n);}
QueryPerformanceCounter((LARGE_INTEGER *)&tail );
cout<<(tail - head)*1000.0/freq<<' ';


//对齐
QueryPerformanceCounter((LARGE_INTEGER *)&head);
for(int i=0;i<10;i++){
A4 = aline(A4,n);}
QueryPerformanceCounter((LARGE_INTEGER *)&tail );
cout<<(tail - head)*1000.0/freq<<' ';

cout<<endl;
}

return 0;}
