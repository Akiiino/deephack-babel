INPUT_FOLDER=data
INTERM_FOLDER=temp

PARALLEL_CORPUS=$(INPUT_FOLDER)/parallel_corpus.txt
CORPUS=$(INPUT_FOLDER)/corpus
INPUT=$(INPUT_FOLDER)/input.txt

MODEL_PREFIX=$(INTERM_FOLDER)/lang-
ENCODED_MONO=$(INTERM_FOLDER)/mono-
ENCODED_PARAL=$(INTERM_FOLDER)/paral-
SPLIT_PARALLEL=$(INTERM_FOLDER)/parallel_split-
FULL=$(INTERM_FOLDER)/full-
EMBEDDED_MONO=$(INTERM_FOLDER)/emb-

ENCODED_INPUT=$(INTERM_FOLDER)/input.txt

CORENLP=corenlp/stanford-corenlp-3.8.0.jar
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
# MUSE=python3 aMUSE/unsupervised_wgan.py --cuda True --src_lang src --tgt_lang tgt --emb_dim 300 --dis_most_frequent 10000 --dis_smooth 0.3 --batch_size 2048  --dis_dropout 0.2 --dis_clip_weights 1 --refinement True --n_iters 25 --exp_path $(INTERM_FOLDER)

encode:
	mkdir -p $(INTERM_FOLDER)

	cp $(CORPUS)1.txt $(CORPUS)-src.txt
	cp $(CORPUS)2.txt $(CORPUS)-tgt.txt
	cut $(PARALLEL_CORPUS) -f1 > $(SPLIT_PARALLEL)src.txt ; \
	cut $(PARALLEL_CORPUS) -f2 > $(SPLIT_PARALLEL)tgt.txt ; \

	for doc in src tgt; do \
		cat $(CORPUS)-$$doc.txt $(SPLIT_PARALLEL)$$doc.txt > $(FULL)$$doc.txt ; \
		CLASSPATH=$(CORENLP) java edu.stanford.nlp.process.PTBTokenizer -preserveLines  $(FULL)$$doc.txt > $(ENCODED_MONO)$$doc.txt ; \
		CLASSPATH=$(CORENLP) java edu.stanford.nlp.process.PTBTokenizer -preserveLines  $(SPLIT_PARALLEL)$$doc.txt > $(ENCODED_PARAL)$$doc.txt ; \
	done

	CLASSPATH=$(CORENLP) java edu.stanford.nlp.process.PTBTokenizer -preserveLines  $(INPUT) > $(ENCODED_INPUT)

	touch $@


embed: encode
	for doc in src tgt; do \
		$(FASTTEXT) -input $(ENCODED_MONO)$$doc.txt -output $(EMBEDDED_MONO)$$doc ; \
	done

	touch $@

parallel_embed: embed
	$(MUSE) --src_emb $(EMBEDDED_MONO)src.vec --tgt_emb $(EMBEDDED_MONO)tgt.vec

	touch $@

clean:
	rm -rf $(INTERM_FOLDER)
	rm -f encode
	rm -f embed
	rm -f parallel_embed
