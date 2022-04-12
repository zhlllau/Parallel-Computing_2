#include<iostream>
#include<arm_neon.h>  
#include <sys/time.h>
#include<time.h>

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




float** Neon_gaosi_part1(float **A, int n){
    //Neon算法——part1
    float32x4_t t1 ;
    float32x4_t t2 ;
    float32x4_t t3 ;

    for (int k=0;k<n;k++){

      //将第k行第k个元素全部变为1
      float tmp[4] = {A[k][k], A[k][k], A[k][k], A[k][k]}; //创建一个浮点数向量A[k][k]
      t1=vld1q_f32(tmp);  //将向量float装载进t1中 {*不对齐形式}

      int j = n-4;
      for (j = n-4; j >= k ; j = j-4)

        {   t2 = vld1q_f32(A[k] + j);
            t3 = vdivq_f32(t2, t1);
            vst1q_f32(A[k]+j, t3);

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

     return A;
}

float** SSE_gaosi_part2(float **A, int n){

    float32x4_t t1 ;
    float32x4_t t2 ;
    float32x4_t t3 ;
    float32x4_t t4 ;

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
           t1=vld1q_f32(tmp); 

           int j;
           for(j=n-4;j>=k;j=j-4){

              
                 t2 = vld1q_f32(A[i] + j);
                 t3 = vld1q_f32(A[k] + j);
                 t4 = vsubq_f32(t2, vmulq_f32(t1, t3));
                 vst1q_f32(A[i]+j, t4);

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
    float32x4_t t1;
    float32x4_t t2;
    float32x4_t t3;
    float32x4_t t4;

    for (int k=0;k<n;k++){

      float tmp[4] = {A[k][k], A[k][k], A[k][k], A[k][k]}; //创建一个浮点数向量A[k][k]
      float32x4_t t1=vld1q_f32(tmp);  //将向量float装载进t1中 {*不对齐形式}

      int j = n-4;
      for (j = n-4; j >= k ; j = j-4)

        {   t2 = vld1q_f32(A[k] + j);
            t3 = vdivq_f32(t2, t1);
            vst1q_f32(A[k]+j, t3);

        }

       if((j+4)!=k){
                for(int i=k; (i<n)&&(i<(j+4)); i++){
                    A[k][i] = A[k][i]/A[k][k];
                }
        }

      //对k+1到第n-1行进行高斯消元
     for(int i=k+1;i<n;i++){

           float tmp[4] = {A[i][k], A[i][k], A[i][k], A[i][k]};
           t1=vld1q_f32(tmp); 
           //t1 = _mm_loadu_ps(tmp);

           int j;
           for(j=n-4;j>=k;j=j-4){

                
                 t2 = vld1q_f32(A[i] + j);
                 t3 = vld1q_f32(A[k] + j);
                 t4 = vsubq_f32(t2, vmulq_f32(t1, t3));
                 vst1q_f32(A[i]+j, t4);

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

int n=8;
for(int n;n<=256;n=n*2){
cout<<"n为"<<n<<endl;
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



struct timeval start;
struct timeval end;


cout<<"串行算法"<<endl;
gettimeofday(&start,NULL);
for(int i=0;i<100;i++)
{A = normal_gaosi(A,n);}
gettimeofday(&end,NULL);
cout<<((long long)end.tv_sec-(long long)start.tv_sec)*1000000+((long long)end.tv_usec-(long long)start.tv_usec)<<endl;//微秒

cout<<"NEON_part1"<<endl;
gettimeofday(&start,NULL);
for(int i=0;i<100;i++){
A2 = normal_gaosi(A2,n);}
gettimeofday(&end,NULL);
cout<<((long long)end.tv_sec-(long long)start.tv_sec)*1000000+((long long)end.tv_usec-(long long)start.tv_usec)<<endl;//微秒


cout<<"NEON_part2"<<endl;
gettimeofday(&start,NULL);
for(int i=0;i<100;i++){
A3 = normal_gaosi(A3,n);
}
gettimeofday(&end,NULL);
cout<<((long long)end.tv_sec-(long long)start.tv_sec)*1000000+((long long)end.tv_usec-(long long)start.tv_usec)<<endl;//微秒


cout<<"NEON_all"<<endl;
gettimeofday(&start,NULL);
for(int i=0;i<100;i++){
A4 = normal_gaosi(A4,n);}
gettimeofday(&end,NULL);
cout<<((long long)end.tv_sec-(long long)start.tv_sec)*1000000+((long long)end.tv_usec-(long long)start.tv_usec)<<endl;//微秒

}
cout<<endl;

return 0;}
