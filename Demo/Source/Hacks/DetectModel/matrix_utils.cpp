#include <math.h>

double sigmoid(double x) {
	return 1 / (1 + exp(-x));
}

void mat_sum(double* A, double* B, double* C, unsigned int N, unsigned int M)
{
	int i, j;
	for (i = 0; i < N; i++)
	{
		for (j = 0; j < M; j++)
		{
			C[i * M + j] = A[i * M + j] + B[i * M + j];
		}
	}
}
void mat_mul(double* A, double* B, double* C, unsigned int N, unsigned int K, unsigned int M)
{
	int i, j, k;
	for (i = 0; i < N; i++)
	{
		for (j = 0; j < M; j++)
		{
			C[i * M + j] = 0.0;
		}
	}
	for (i = 0; i < N; i++)
	{
		for (j = 0; j < K; j++)
		{
			for (k = 0; k < M; k++)
			{
				C[i * M + k] += A[i * K + j] * B[j * M + k];
			}
		}
	}
}
void mat_emul(double* A, double* B, double* C, unsigned int N, unsigned int M)
{
	int i, j;
	for (i = 0; i < N; i++)
	{
		for (j = 0; j < M; j++)
		{
			C[i * M + j] = A[i * M + j] * B[i * M + j];
		}
	}
}
void mat_sigmoid(double* src, double* dst, unsigned int N, unsigned int M)
{
	int i, j;
	for (i = 0; i < N; i++)
	{
		for (j = 0; j < M; j++)
		{
			dst[i * M + j] = sigmoid(src[i * M + j]);
		}
	}
}
void mat_tanh(double* src, double* dst, unsigned int N, unsigned int M)
{
	int i, j;
	for (i = 0; i < N; i++)
	{
		for (j = 0; j < M; j++)
		{
			dst[i * M + j] = tanhf(src[i * M + j]);
		}
	}
}

void mat_1mz(double* src, double* dst, unsigned int N, unsigned int M)
{
	int i, j;
	for (i = 0; i < N; i++)
	{
		for (j = 0; j < M; j++)
		{
			dst[i * M + j] = 1 - src[i * M + j];
		}
	}
}