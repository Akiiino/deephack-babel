INPUT_FOLDER=data
INTERM_FOLDER=temp

VOCAB_SIZE=20000

HAS_OUTPUT=0
OUTPUT=$(INPUT_FOLDER)/output.txt
TOKENIZED_OUTPUT=$(INTERM_FOLDER)/output.txt

PARALLEL_CORPUS=$(INPUT_FOLDER)/parallel_corpus.txt
CORPUS=$(INPUT_FOLDER)/corpus
INPUT=$(INPUT_FOLDER)/input.txt

MODEL_PREFIX=$(INTERM_FOLDER)/lang-
TOKENIZED_MONO=$(INTERM_FOLDER)/mono-
TOKENIZED_PARAL=$(INTERM_FOLDER)/paral-
SPLIT_PARALLEL=$(INTERM_FOLDER)/parallel_split-
FULL=$(INTERM_FOLDER)/full-
EMBEDDED_MONO=$(INTERM_FOLDER)/emb-

TOKENIZED_INPUT=$(INTERM_FOLDER)/input.txt

TRAIN_TOKENIZER=spm_train --vocab_size=$(VOCAB_SIZE) --model_type=unigram --num_threads 4
TOKENIZER=spm_encode --output_format=piece
FASTTEXT=fastText/fasttext skipgram -dim 300 -epoch 5 -thread 4
MUSE=python3 MUSE/unsupervised.py \
	--cuda True \
	--src_lang src \
	--tgt_lang tgt \
	--emb_dim 300 \
	--dis_most_frequent 10000 \
	--dis_smooth 0.3 \
	--batch_size 2048  \
	--dis_dropout 0.5 \
	--refinement True \
	--n_iters 25 \
	--n_epochs 15 \
	--verbose 1 \
	--exp_path $(INTERM_FOLDER)

res1= $(shell wc -l $(CORPUS)1.txt | cut -d " " -f1)
res2= $(shell wc -l $(CORPUS)2.txt | cut -d " " -f1)

encode:
	if [ $(res1) -lt 10 ] || [ $(res2) -lt 10 ]; then\
		echo "CUTTING" ;\
		VOCAB_SIZE:=10 ;\
	fi;

	mkdir -p $(INTERM_FOLDER)

	cp $(CORPUS)1.txt $(CORPUS)-src.txt
	cp $(CORPUS)2.txt $(CORPUS)-tgt.txt
	cut $(PARALLEL_CORPUS) -f1 > $(SPLIT_PARALLEL)src.txt ; \
	cut $(PARALLEL_CORPUS) -f2 > $(SPLIT_PARALLEL)tgt.txt ; \

	for doc in src tgt; do \
		cat $(CORPUS)-$$doc.txt $(SPLIT_PARALLEL)$$doc.txt > $(FULL)$$doc.txt ; \
		$(TRAIN_TOKENIZER)  --model_prefix=$(MODEL_PREFIX)$$doc --input=$(FULL)$$doc.txt ; \
		$(TOKENIZER) --model=$(MODEL_PREFIX)$$doc.model < $(FULL)$$doc.txt > $(TOKENIZED_MONO)$$doc.txt ; \
		$(TOKENIZER) --model=$(MODEL_PREFIX)$$doc.model < $(SPLIT_PARALLEL)$$doc.txt > $(TOKENIZED_PARAL)$$doc.txt ; \
	done

	$(TOKENIZER) --model=$(MODEL_PREFIX)src.model $(INPUT) > $(TOKENIZED_INPUT)

	touch $@


embed: encode
	for doc in src tgt; do \
		$(FASTTEXT) -input $(TOKENIZED_MONO)$$doc.txt -output $(EMBEDDED_MONO)$$doc ; \
	done

	touch $@

parallel_embed: embed
	$(MUSE) --src_emb $(EMBEDDED_MONO)src.vec --tgt_emb MUSE/emb-stand.txt
	$(MUSE) --src_emb $(EMBEDDED_MONO)tgt.vec --tgt_emb MUSE/emb-stand.txt

	touch $@

train_transformer: encode
	python3 transformer/preprocess.py -train_src $(INTERM_FOLDER)/paral-src.txt -train_tgt $(INTERM_FOLDER)/paral-tgt.txt -valid_src $(INTERM_FOLDER)/paral-src.txt -valid_tgt $(INTERM_FOLDER)/paral-tgt.txt -save_data data.svd

	python3 transformer/train.py -data data.svd -save_model trained -save_mode best -proj_share_weight

	python3 transformer/translate.py -model trained.chkpt -vocab data.svd -src $(INTERM_FOLDER)/input.txt

	spm_decode --model=$(MODEL_PREFIX)tgt.model --input_format=piece < pred.txt > /output/output.

clean:
	rm -rf $(INTERM_FOLDER)
	rm -f encode
	rm -f embed
	rm -f parallel_embed
	rm -f train_transformer
