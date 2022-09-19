#include "SGX_MDL_t.h"

#include "sgx_trts.h"

#include "GRU.h"
#include "mini-gmp.h"
#include "sha256.h"

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

struct GRU* gru = NULL;

struct player_score
{
	char* username;
	double max_score;
	struct player_score* nxt;
};

struct player_score* plst = NULL;

const char* s_prime = "a532dbaaa83d5e2657aa7829d74df75ebcad3626a9debe75c4456ae659f2c55bb0c895a713fa4ce1778878ea173d51fa693e965ce02ac5b7892b26eede0495ca26473567f943301abd66e9b207c6edadb1798f77300435b1e39c31617ebb9271cfdf21d6191de1a6ac24a6bc7aac51e11585bb665218a6eb5a0131082fc72871";
mpz_t n;
mpz_t e;
unsigned char kk[101];

void init_sgx(uint8_t* buf)
{
	int err;
	mpz_t k;
	mpz_t enc;

	mpz_init(n);
	mpz_init(e); mpz_set_si(e, 65537);


	err = mpz_set_str(n, s_prime, 16);

	for (int i = 0; i < 100; i++)
	{
		sgx_read_rand(&kk[i], 1);
		kk[i] %= 10;
		kk[i] += '0';
	}
	kk[100] = 0;

	mpz_init(k);
	err = mpz_set_str(k, kk, 10);

	mpz_init(enc);
	mpz_powm(enc, k, e, n);
	
	char *tmp = mpz_get_str(0, 16, enc);
	memcpy(buf, tmp, 256);
}

// input preprocessed WIN_SIZE+1 frames * 3 features
void process(char *username, double* input, char* output_buf, size_t size, size_t *res_size)
{
	double output[IPT_SIZE] = { 0.0, };
	if (gru == NULL) gru = init_gru();
	run_network(gru, input, output);

	// L1 Norm
	int i;
	double norm = 0.0;
	for (i = 0; i < IPT_SIZE; i++)
	{
		double diff = fabs(output[i] - input[WIN_SIZE * IPT_SIZE + i]);
		norm += diff;
	}
	norm /= IPT_SIZE;

	struct player_score* it = plst;
	while (it != NULL)
	{
		if (!strcmp(it->username, username))
			break;
		it = it->nxt;
	}
	if (it == NULL)
	{
		struct player_score* nw = malloc(sizeof(struct player_score));
		nw->username = strndup(username, strlen(username));
		nw->max_score = 0;
		nw->nxt = plst;
		plst = nw;
		it = nw;
	}

	if (norm > it->max_score)
	{
		it->max_score = norm;

		char buf[0x400];
		char sign[0x20];
		snprintf(buf, 0x400 - 2 - 0x20, "%s/%lf", username, it->max_score);

		SHA256_CTX ctx;
		sha256_init(&ctx);
		sha256_update(&ctx, (const BYTE*)buf, strlen(buf));
		//sha256_update(&ctx, (const BYTE*)key, 32);
		sha256_final(&ctx, (BYTE*)sign);
		
		*((unsigned short*)output_buf) = (unsigned short)strlen(buf);
		memcpy(output_buf + 2, buf, strlen(buf));
		memcpy(output_buf + 2 + strlen(buf), sign, 32);

		*res_size = 2 + strlen(buf) + 32;
	}
	else
	{
		*res_size = 0;
	}

}