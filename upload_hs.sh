#!/usr/bin/env bash
scp -i ucu.pem ./preprocessed/hs/train/ccg_train.pth ubuntu@188.163.246.10:~/diploma/my/preprocessed/hs/train
scp -i ucu.pem ./preprocessed/hs/dev/ccg_dev.pth ubuntu@188.163.246.10:~/diploma/my/preprocessed/hs/dev
scp -i ucu.pem ./preprocessed/hs/test/ccg_test.pth ubuntu@188.163.246.10:~/diploma/my/preprocessed/hs/test

scp -i ucu.pem ./preprocessed/hs/train/pcfg_train.pth ubuntu@188.163.246.10:~/diploma/my/preprocessed/hs/train
scp -i ucu.pem ./preprocessed/hs/dev/pcfg_dev.pth ubuntu@188.163.246.10:~/diploma/my/preprocessed/hs/dev
scp -i ucu.pem ./preprocessed/hs/test/pcfg_test.pth ubuntu@188.163.246.10:~/diploma/my/preprocessed/hs/test

scp -i ucu.pem ./preprocessed/hs/train/dependency_train.pth ubuntu@188.163.246.10:~/diploma/my/preprocessed/hs/train
scp -i ucu.pem ./preprocessed/hs/dev/dependency_dev.pth ubuntu@188.163.246.10:~/diploma/my/preprocessed/hs/dev
scp -i ucu.pem ./preprocessed/hs/test/dependency_test.pth ubuntu@188.163.246.10:~/diploma/my/preprocessed/hs/test

scp -i ucu.pem ./preprocessed/hs/word_embeddings.pth ubuntu@188.163.246.10:~/diploma/my/preprocessed/hs