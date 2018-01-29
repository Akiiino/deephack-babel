spm_train --input=corpus1.txt --model_prefix=lang1 --vocab_size=50000 --model_type=word --num_threads 4
spm_train --input=corpus2.txt --model_prefix=lang2 --vocab_size=50000 --model_type=word --num_threads 4
spm_encode --model=lang1.model --output_format=piece --output monolng1.txt corpus1.txt
spm_encode --model=lang2.model --output_format=piece --output monolng2.txt corpus2.txt
cut parallel_corpus.txt -f1 > paral1_pre.txt
cut parallel_corpus.txt -f2 > paral2_pre.txt
spm_encode --model=lang1.model --output_format=piece --output paral1.txt paral1_pre.txt
spm_encode --model=lang2.model --output_format=piece --output paral2.txt paral2_pre.txt
spm_encode --model=lang1.model --output_format=piece --output inp.txt input.txt
