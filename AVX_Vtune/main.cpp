#include<iostream>
#include <immintrin.h>//AVX512
#include<windows.h>
#include <immintrin.h>//AVX
#include <xmmintrin.h>

using namespace std;


float** normal_gaosi(float **A, int n){
	//�����㷨���岿��
for (int k=0;k<n;k++){

      //����k�е�k��Ԫ��ȫ����Ϊ1
      for(int j=k+1;j<n;j++)
           {
			A[k][j] = A[k][j]/A[k][k];
	       }
      A[k][k]=1.0;


      //��k+1����n-1�н��и�˹��Ԫ
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
    //SSE�㷨����part1
    __m256 t1, t2, t3;

    for (int k=0;k<n;k++){

      //����k�е�k��Ԫ��ȫ����Ϊ1
      float tmp[8] = {A[k][k], A[k][k], A[k][k], A[k][k], A[k][k],A[k][k],A[k][k],A[k][k]}; //����һ������������A[k][k]
      t1 = _mm256_loadu_ps(tmp);  //������floatװ�ؽ�t1�� {*��������ʽ}

      int j = n-8;
      for (j = n-8; j >= k ; j = j-8)

        {
            t2 = _mm256_loadu_ps(A[k] + j);
            t3 = _mm256_div_ps(t2, t1);
            _mm256_storeu_ps(A[k] + j, t3);

        }

       if((j+8)!=k){
                for(int i=k; (i<n)&&(i<(j+8)); i++){
                    A[k][i] = A[k][i]/A[k][k];
                }
        }

      //��k+1����n-1�н��и�˹��Ԫ

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

     __m256 t1, t2, t3, t4;
    //SSE�㷨����part2
    for (int k=0;k<n;k++){

      //����k�е�k��Ԫ��ȫ����Ϊ1-����
      for(int j=k+1;j<n;j++)
           {
			A[k][j] = A[k][j]/A[k][k];

	        }
      A[k][k]=1.0;

      //��k+1����n-1�н��и�˹��Ԫ
      //����SSE-part2

      for(int i=k+1;i<n;i++){

           float tmp[8] = {A[i][k], A[i][k], A[i][k], A[i][k],A[i][k], A[i][k], A[i][k], A[i][k]};
           t1 = _mm256_loadu_ps(tmp);

           int j;
           for(j=n-8;j>=k;j=j-8){

                 //A[i][j] = A[i][j]-A[i][k]*A[k][j];
                 t2 = _mm256_loadu_ps(A[i] + j);
                 t3 = _mm256_loadu_ps(A[k] + j);
                 t4 = _mm256_sub_ps(t2, _mm256_mul_ps(t1, t3));
                 _mm256_storeu_ps(A[i]+j, t4);

                                  }
           if((j+8)!=k)
      {
          for(int s=k; s<(j+8);s++)
              A[i][s] = A[i][s]-A[i][k]*A[k][s];

      }
           }
     }
     return A;

}


float** SSE_gaosi_all(float **A, int n){
    //SSE�㷨����all
    __m256 t1, t2, t3, t4;


    for (int k=0;k<n;k++){
      float tmp[8] = {A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k]}; //����һ������������A[k][k]
      t1 = _mm256_loadu_ps(tmp);  //������floatװ�ؽ�t1�� {*��������ʽ}

      int j = n-8;
      for(j=n-8;j>=k;j = j-8)
           {
			//��A[k][j] = A[k][j]/A[k][k]���л�;
            t2 = _mm256_loadu_ps(A[k]+j);   //����k�е�j~j+4��Ԫ��װ�ؽ���
            t3 = _mm256_div_ps(t2, t1); //������
            _mm256_storeu_ps(A[k]+j, t3);   //���������װ�ػ�ȥ

            }
      //�����ܱ�4��������
        if((j+8)!=k){
                for(int i=k; (i<n)&&(i<(j+8)); i++){
                    A[k][i] = A[k][i]/A[k][k];
                }
        }


      //��k+1����n-1�н��и�˹��Ԫ
      for(int i=k+1;i<n;i++){

           float tmp[8] = {A[i][k], A[i][k], A[i][k], A[i][k],A[i][k], A[i][k], A[i][k], A[i][k]};
           t1 = _mm256_loadu_ps(tmp);

           int j = n-8;
           for(j=n-8;j>=k;j=j-8){
                 //A[i][j] = A[i][j]-A[i][k]*A[k][j];
                 t2 = _mm256_loadu_ps(A[i] + j);
                 t3 = _mm256_loadu_ps(A[k] + j);
                 t4 = _mm256_sub_ps(t2, _mm256_mul_ps(t1, t3));
                 _mm256_storeu_ps(A[i]+j, t4);}

           if((j+8)!=k)
      {
          for(int s=k; s<(j+8);s++)
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




int main(){

for(int n=8;n<=1024;n=n*2){
cout<<"nΪ��"<<n<<endl;
//���ɾ���
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
cout<<"��ͨ��˹����Time: "<<(tail - head)*1000.0/freq<<"ms"<<endl;


QueryPerformanceCounter((LARGE_INTEGER *)&head);
for(int i=0;i<10;i++){
A2 = SSE_gaosi_part1(A2,n);
}
QueryPerformanceCounter((LARGE_INTEGER *)&tail );
cout<<"AVX��part1��Time: "<<(tail - head)*1000.0/freq<<"ms"<<endl;


QueryPerformanceCounter((LARGE_INTEGER *)&head);
for(int i=0;i<10;i++){
A3 = SSE_gaosi_part2(A3,n);
}
QueryPerformanceCounter((LARGE_INTEGER *)&tail );
cout<<"AVX��part2��Time: "<<(tail - head)*1000.0/freq<<"ms"<<endl;


QueryPerformanceCounter((LARGE_INTEGER *)&head);
for(int i=0;i<10;i++){
A4 = SSE_gaosi_all(A4,n);
}
QueryPerformanceCounter((LARGE_INTEGER *)&tail );
cout<<"AVX��all��Time: "<<(tail - head)*1000.0/freq<<"ms"<<endl;

cout<<endl;
}
return 0;}
