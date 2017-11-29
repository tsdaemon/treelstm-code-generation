#!/usr/bin/env bash
# scp -i ucu.pem ./preprocessed/hs/train/train.pth ubuntu@188.163.246.10:~/diploma/my/preprocessed/hs/train
# scp -i ucu.pem ./preprocessed/hs/dev/dev.pth ubuntu@188.163.246.10:~/diploma/my/preprocessed/hs/dev
# scp -i ucu.pem ./preprocessed/hs/test/test.pth ubuntu@188.163.246.10:~/diploma/my/preprocessed/hs/test
scp -i ucu.pem ./preprocessed/hs/word_embeddings.pth ubuntu@188.163.246.10:~/diploma/my/preprocessed/hs