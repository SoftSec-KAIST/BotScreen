#include "GRU.h"

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
		struct player_score* nw = (struct player_score *)malloc(sizeof(struct player_score));
		nw->username = strdup(username);
		nw->max_score = 0;
		nw->nxt = plst;
		plst = nw;
		it = nw;
	}

	// if (norm > it->max_score)
	{
		it->max_score = norm;

		char *buf = (char *)malloc(size);
		snprintf(buf, size, "%s/%lf", username, it->max_score);

		*((unsigned short*)output_buf) = (unsigned short)strlen(buf);
		memcpy(output_buf + 2, buf, strlen(buf));

		*res_size = 2 + strlen(buf);
	}
}

// FIXME : no memory releases for Model
// when do we do that? match_end? cs_win_panel_match? free and realloc when cs_match_end_restart?