INPUT_FOLDER=data
INTERM_FOLDER=temp

VOCAB_SIZE=20000

PARALLEL_CORPUS=$(INPUT_FOLDER)/parallel_corpus.txt
CORPUS=$(INPUT_FOLDER)/corpus
INPUT=$(INPUT_FOLDER)/input.txt

MODEL_PREFIX=$(INTERM_FOLDER)/lang
ENCODED_MONO=$(INTERM_FOLDER)/mono
ENCODED_PARAL=$(INTERM_FOLDER)/paral
SPLIT_PARALLEL=$(INTERM_FOLDER)/parallel_split
FULL=$(INTERM_FOLDER)/full
EMBEDDED_MONO=$(INTERM_FOLDER)/emb

ENCODED_INPUT=$(INTERM_FOLDER)/input.txt

FASTTEXT=fastText/fasttext
MUSE=python3 MUSE_clean/unsupervised.py --cuda True --src_lang 1 --tgt_lang 2 --emb_dim 100 --dis_most_frequent 10000 --dis_smooth 0.15

encode:
	mkdir -p $(INTERM_FOLDER)

	for doc in 1 2; do \
		cut $(PARALLEL_CORPUS) -f$$doc > $(SPLIT_PARALLEL)$$doc.txt ; \
		cat $(CORPUS)$$doc.txt $(SPLIT_PARALLEL)$$doc.txt > $(FULL)$$doc.txt ; \
		spm_train --input=$(FULL)$$doc.txt --model_prefix=$(MODEL_PREFIX)$$doc --vocab_size=$(VOCAB_SIZE) --model_type=unigram --num_threads 4 ; \
		spm_encode --model=$(MODEL_PREFIX)$$doc.model --output_format=piece --output $(ENCODED_MONO)$$doc.txt $(FULL)$$doc.txt ; \
		spm_encode --model=$(MODEL_PREFIX)$$doc.model --output_format=piece --output $(ENCODED_PARAL)$$doc.txt $(SPLIT_PARALLEL)$$doc.txt ; \
	done

	spm_encode --model=$(MODEL_PREFIX)1.model --output_format=piece --output $(ENCODED_INPUT) $(INPUT)\

	touch $@

embed: encode
	for doc in 1 2; do \
		$(FASTTEXT) skipgram -input $(ENCODED_MONO)$$doc.txt -output $(EMBEDDED_MONO)$$doc ; \
	done

	touch $@

parallel_embed: embed
	$(MUSE) --src_emb $(EMBEDDED_MONO)1.vec --tgt_emb $(EMBEDDED_MONO)2.vec

	touch $@

clean:
	rm -rf $(INTERM_FOLDER)
	rm -f encode
	rm -f embed
	rm -f parallel_embed
