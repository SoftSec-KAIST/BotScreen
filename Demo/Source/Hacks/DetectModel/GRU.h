#ifndef _GRU_H_
#define _GRU_H_

#define WIN_SIZE 20
#define IPT_SIZE 3

struct GRU_cell
{
	unsigned int in_size;
	unsigned int out_size;
	double* W_ir;
	double* W_hr;
	double* W_iz;
	double* W_hz;
	double* W_in;
	double* W_hn;
	double* B_ir;
	double* B_hr;
	double* B_iz;
	double* B_hz;
	double* B_in;
	double* B_hn;
};

struct GRU
{
	unsigned int input_size;
	unsigned int hidden_size;
	unsigned int frame_size;
	struct GRU_cell* layer[3];
	struct GRU_cell* layer_reverse[3];
};

struct GRU_cell* init_gru_cell(unsigned int in, unsigned int out, double* Wih, double* Whh, double* Bih, double* Bhh);
void free_gru_cell(struct GRU_cell* ptr);
void run_gru_cell(struct GRU_cell* cell, double* input, double* h, double* output);
struct GRU* init_gru();
void free_gru(struct GRU* ptr);
void run_gru(struct GRU* layer, double* input, double* output);
void run_network(struct GRU* gru, double* input, double* output);
#endif
