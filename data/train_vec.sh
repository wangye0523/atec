
word2vec="/home/rocky/dl/word2vec/trunk/word2vec"
train_file="./seg_result.txt"

time ${word2vec} -train $train_file -output vectors.txt -cbow 1 -size 300 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 10 -binary 0 -iter 15 -save-vocab vocab.txt
