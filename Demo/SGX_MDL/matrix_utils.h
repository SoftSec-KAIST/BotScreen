#ifndef _MATRIX_UTILS_H_
#define _MATRIX_UTILS_H_
void mat_sum(double* A, double* B, double* C, unsigned int N, unsigned int M);
void mat_mul(double* A, double* B, double* C, unsigned int N, unsigned int K, unsigned int M);
void mat_emul(double* A, double* B, double* C, unsigned int N, unsigned int M);
void mat_sigmoid(double* src, double* dst, unsigned int N, unsigned int M);
void mat_tanh(double* src, double* dst, unsigned int N, unsigned int M);
void mat_1mz(double* src, double* dst, unsigned int N, unsigned int M);
#endif