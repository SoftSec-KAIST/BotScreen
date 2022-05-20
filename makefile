### python version ###
py := python3			# python3 command

### experiment arguments ###
gpu := 0				# gpu id
seed := 3				# random seed

### sgru model arguments ###
n_hidden := 64			# number of hidden units
n_layer := 3			# number of stacked layers
bat_size := 64			# batch size
n_epoch := 64			# number of training epochs
win_size := 20			# rnn window size
stride := 2				# stride for training events
duration := 10			# time duration for relevant events

### cross validation ###
n_split := 7			# number of splits in k-fold cross validation
split := -1				# split to experiment on

### benchmark results ###
save_result := False	# save bench results to a tsv file

### all arguments ###
argnames := n_hidden n_layer bat_size n_epoch win_size stride duration seed gpu n_split split save_result
args := $(foreach a,$(argnames), --$(a) $($(a)))

### make targets ###
%: %.py
	@$(py) -u $^ $(args)

### dummy makefile target used for preprocessing ###
exps := exp_1 exp_2 exp_3 exp_4
preprocess_all:
	@$(foreach e, $(exps),\
		$(foreach g, $(shell seq 1 10),\
			$(py) -u preprocess.py $(exp_name) --game $(g);))

##### clean everything #####
clean:
	@find . -name '__pycache__' -type d -exec rm -rf {} +