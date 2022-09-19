#include<malloc.h>
#include<string.h>
#include"matrix_utils.h"
#include"state.h"
#include"GRU.h"

struct GRU_cell* init_gru_cell(unsigned int in, unsigned int out, double* Wih, double* Whh, double* Bih, double* Bhh)
{
	struct GRU_cell* res = (struct GRU_cell*)malloc(sizeof(struct GRU_cell));
	res->in_size = in;
	res->out_size = out;

	res->W_hr = (double*)malloc(sizeof(double) * out * out);
	res->W_hz = (double*)malloc(sizeof(double) * out * out);
	res->W_hn = (double*)malloc(sizeof(double) * out * out);

	res->W_ir = (double*)malloc(sizeof(double) * out * in);
	res->W_iz = (double*)malloc(sizeof(double) * out * in);
	res->W_in = (double*)malloc(sizeof(double) * out * in);

	res->B_ir = (double*)malloc(sizeof(double) * out);
	res->B_iz = (double*)malloc(sizeof(double) * out);
	res->B_in = (double*)malloc(sizeof(double) * out);

	res->B_hr = (double*)malloc(sizeof(double) * out);
	res->B_hz = (double*)malloc(sizeof(double) * out);
	res->B_hn = (double*)malloc(sizeof(double) * out);

	memcpy(res->W_ir, Wih, sizeof(double) * out * in);
	memcpy(res->W_iz, Wih + 1 * out * in, sizeof(double) * out * in);
	memcpy(res->W_in, Wih + 2 * out * in, sizeof(double) * out * in);

	memcpy(res->W_hr, Whh, sizeof(double) * out * out);
	memcpy(res->W_hz, Whh + 1 * out * out, sizeof(double) * out * out);
	memcpy(res->W_hn, Whh + 2 * out * out, sizeof(double) * out * out);

	memcpy(res->B_ir, Bih, sizeof(double) * out);
	memcpy(res->B_iz, Bih + 1 * out, sizeof(double) * out);
	memcpy(res->B_in, Bih + 2 * out, sizeof(double) * out);

	memcpy(res->B_hr, Bhh, sizeof(double) * out);
	memcpy(res->B_hz, Bhh + 1 * out, sizeof(double) * out);
	memcpy(res->B_hn, Bhh + 2 * out, sizeof(double) * out);

	return res;
}

void free_gru_cell(struct GRU_cell* ptr)
{
	free(ptr->W_ir);
	free(ptr->W_iz);
	free(ptr->W_in);
	free(ptr->W_hr);
	free(ptr->W_hz);
	free(ptr->W_hn);
	free(ptr->B_ir);
	free(ptr->B_iz);
	free(ptr->B_in);
	free(ptr->B_hr);
	free(ptr->B_hz);
	free(ptr->B_hn);
	free(ptr);
}

void run_gru_cell(struct GRU_cell* cell, double* input, double* h, double* output)
{
	double* tmp1 = (double*)malloc(sizeof(double) * cell->out_size);
	double* tmp2 = (double*)malloc(sizeof(double) * cell->out_size);

	// calculate r
	// r = sigmoid(W_ir @ input + B_ir + W_hr @ h + B_hr)
	double* r = (double*)malloc(sizeof(double) * cell->out_size);

	mat_mul(cell->W_ir, input, tmp1, cell->out_size, cell->in_size, 1);
	mat_sum(tmp1, cell->B_ir, tmp1, cell->out_size, 1);
	mat_mul(cell->W_hr, h, tmp2, cell->out_size, cell->out_size, 1);
	mat_sum(tmp1, tmp2, tmp1, cell->out_size, 1);
	mat_sum(tmp1, cell->B_hr, tmp1, cell->out_size, 1);
	mat_sigmoid(tmp1, r, cell->out_size, 1);

	// calculate z
	// z = sigmoid(W_iz @ input + B_iz + W_hz @ h + B_hz)
	double* z = (double*)malloc(sizeof(double) * cell->out_size);

	mat_mul(cell->W_iz, input, tmp1, cell->out_size, cell->in_size, 1);
	mat_sum(tmp1, cell->B_iz, tmp1, cell->out_size, 1);
	mat_mul(cell->W_hz, h, tmp2, cell->out_size, cell->out_size, 1);
	mat_sum(tmp1, tmp2, tmp1, cell->out_size, 1);
	mat_sum(tmp1, cell->B_hz, tmp1, cell->out_size, 1);
	mat_sigmoid(tmp1, z, cell->out_size, 1);

	// calculate n
	// n = tanh(W_in @ input + B_in + r * (W_hn @ h + B_hn))
	double* n = (double*)malloc(sizeof(double) * cell->out_size);

	mat_mul(cell->W_in, input, tmp1, cell->out_size, cell->in_size, 1);
	mat_sum(tmp1, cell->B_in, tmp1, cell->out_size, 1);
	mat_mul(cell->W_hn, h, tmp2, cell->out_size, cell->out_size, 1);
	mat_sum(tmp2, cell->B_hn, tmp2, cell->out_size, 1);
	mat_emul(r, tmp2, tmp2, cell->out_size, 1);
	mat_sum(tmp1, tmp2, tmp1, cell->out_size, 1);
	mat_tanh(tmp1, n, cell->out_size, 1);

	// calculate output
	// output = (1 - z) * n + z * h
	mat_1mz(z, tmp1, cell->out_size, 1);
	mat_emul(tmp1, n, tmp1, cell->out_size, 1);
	mat_emul(z, h, tmp2, cell->out_size, 1);
	mat_sum(tmp1, tmp2, output, cell->out_size, 1);

	free(tmp1);
	free(tmp2);
	free(r);
	free(z);
	free(n);
}

struct GRU* init_gru()
{
	struct GRU* res = (struct GRU*)malloc(sizeof(struct GRU));

	res->input_size = IPT_SIZE;
	res->hidden_size = 64;
	res->frame_size = WIN_SIZE;

	res->layer[0] = init_gru_cell(res->input_size, res->hidden_size, weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0);
	res->layer_reverse[0] = init_gru_cell(res->input_size, res->hidden_size, weight_ih_l0_reverse, weight_hh_l0_reverse, bias_ih_l0_reverse, bias_hh_l0_reverse);
    res->layer[1] = init_gru_cell(res->hidden_size * 2, res->hidden_size, weight_ih_l1, weight_hh_l1, bias_ih_l1, bias_hh_l1);
	res->layer_reverse[1] = init_gru_cell(res->hidden_size * 2, res->hidden_size, weight_ih_l1_reverse, weight_hh_l1_reverse, bias_ih_l1_reverse, bias_hh_l1_reverse);
	res->layer[2] = init_gru_cell(res->hidden_size * 2, res->hidden_size, weight_ih_l2, weight_hh_l2, bias_ih_l2, bias_hh_l2);
	res->layer_reverse[2] = init_gru_cell(res->hidden_size * 2, res->hidden_size, weight_ih_l2_reverse, weight_hh_l2_reverse, bias_ih_l2_reverse, bias_hh_l2_reverse);

	return res;
}

void free_gru(struct GRU* ptr)
{
	free_gru_cell(ptr->layer[0]);
	free_gru_cell(ptr->layer_reverse[0]);
	free_gru_cell(ptr->layer[1]);
	free_gru_cell(ptr->layer_reverse[1]);
	free_gru_cell(ptr->layer[2]);
	free_gru_cell(ptr->layer_reverse[2]);
	free(ptr);
}

void run_gru(struct GRU* layer, double* input, double* output)
{
	int i, j;

	double* res = (double*)malloc(sizeof(double) * layer->hidden_size * 2 * layer->frame_size);
	double* ipt = input;
	double* h = NULL;

	for (i = 0; i < 3; i++)
	{
		// foward
		h = (double*)malloc(sizeof(double) * layer->hidden_size);
		for (j = 0; j < layer->hidden_size; j++) h[j] = 0.0;

		for (j = 0; j < layer->frame_size; j++)
		{
			run_gru_cell(layer->layer[i], ipt + ((i == 0) ? layer->input_size * j : layer->hidden_size * 2 * j), h, res + layer->hidden_size * 2 * j);
			if (j == 0) free(h);
			h = res + layer->hidden_size * 2 * j;
		}

		// backward
		h = (double*)malloc(sizeof(double) * layer->hidden_size);
		for (j = 0; j < layer->hidden_size; j++) h[j] = 0.0;

		for (j = layer->frame_size - 1; j >= 0; j--)
		{
			run_gru_cell(layer->layer_reverse[i], ipt + ((i == 0) ? layer->input_size * j : layer->hidden_size * 2 * j), h, res + layer->hidden_size * (2 * j + 1));
			if (j == layer->frame_size - 1) free(h);
			h = res + layer->hidden_size * (2 * j + 1);
		}

		if (i != 0) free(ipt);
		if (i == 2) break;
		
		// dropout
		char rand;
		for (j = 0; j < layer->hidden_size * 2 * layer->frame_size; j++)
		{
			sgx_read_rand((unsigned char*)&rand, 1);
			if (rand % 10 == 0)
			{
				res[j] = 0.0;
			}
		}
		
		ipt = res;
		res = (double*)malloc(sizeof(double) * layer->hidden_size * 2 * layer->frame_size);
	}
	// return only last frame
	memcpy(output, res + layer->hidden_size * 2 * (layer->frame_size - 1), sizeof(double) * layer->hidden_size * 2);
	free(res);
}

void run_network(struct GRU* gru, double* input, double* output)
{
	double tmp[128];
	double buf[IPT_SIZE];

	// gru layer
	run_gru(gru, input, tmp);
	// fc layer
	mat_mul(fc_w, tmp, buf, IPT_SIZE, 128, 1);
	mat_sum(buf, fc_b, buf, IPT_SIZE, 1);
	memcpy(output, buf, sizeof(double) * IPT_SIZE);
}