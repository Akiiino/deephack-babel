INPUT_FOLDER=data
INTERM_FOLDER=temp

VOCAB_SIZE=20

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

CORENLP=corenlp/stanford-corenlp-3.8.0.jar
FASTTEXT=fastText/fasttext skipgram -dim 300 -epoch 5
MUSE=python3 MUSE_clean/unsupervised.py --cuda True --src_lang 1 --tgt_lang 2 --emb_dim 300 --dis_most_frequent 10000 --dis_smooth 0.3 --batch_size 512  --dis_dropout 0.5 --exp_path $(INTERM_FOLDER)

encode_old:
	mkdir -p $(INTERM_FOLDER)

	for doc in 1 2; do \
		cut $(PARALLEL_CORPUS) -f$$doc > $(SPLIT_PARALLEL)$$doc.txt ; \
		cat $(CORPUS)$$doc.txt $(SPLIT_PARALLEL)$$doc.txt > $(FULL)$$doc.txt ; \
		spm_train --input=$(FULL)$$doc.txt --model_prefix=$(MODEL_PREFIX)$$doc --vocab_size=$(VOCAB_SIZE) --model_type=word --num_threads 4 ; \
		spm_encode --model=$(MODEL_PREFIX)$$doc.model --output_format=piece --output $(ENCODED_MONO)$$doc.txt $(FULL)$$doc.txt ; \
		spm_encode --model=$(MODEL_PREFIX)$$doc.model --output_format=piece --output $(ENCODED_PARAL)$$doc.txt $(SPLIT_PARALLEL)$$doc.txt ; \
	done

	spm_encode --model=$(MODEL_PREFIX)1.model --output_format=piece --output $(ENCODED_INPUT) $(INPUT)

	touch $@

encode:
	mkdir -p $(INTERM_FOLDER)

	for doc in 1 2; do \
		cut $(PARALLEL_CORPUS) -f$$doc > $(SPLIT_PARALLEL)$$doc.txt ; \
		cat $(CORPUS)$$doc.txt $(SPLIT_PARALLEL)$$doc.txt > $(FULL)$$doc.txt ; \
		CLASSPATH=$(CORENLP) java edu.stanford.nlp.process.PTBTokenizer -preserveLines  $(FULL)$$doc.txt > $(ENCODED_MONO)$$doc.txt ; \
		CLASSPATH=$(CORENLP) java edu.stanford.nlp.process.PTBTokenizer -preserveLines  $(SPLIT_PARALLEL)$$doc.txt > $(ENCODED_PARAL)$$doc.txt ; \
	done

	CLASSPATH=$(CORENLP) java edu.stanford.nlp.process.PTBTokenizer -preserveLines  $(INPUT) > $(ENCODED_INPUT)

	touch $@


embed: encode
	for doc in 1 2; do \
		$(FASTTEXT) -input $(ENCODED_MONO)$$doc.txt -output $(EMBEDDED_MONO)$$doc ; \
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
