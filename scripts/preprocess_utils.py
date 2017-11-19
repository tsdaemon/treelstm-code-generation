import os
import re
import platform
from tqdm import tqdm
import torch
from natural_lang.vocab import Vocab

base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
lib_dir = os.path.join(base_dir, 'lib')
data_dir = os.path.join(base_dir, 'data')

delimiter = ';' if platform.system() == 'Windows' else ':'

classpath = delimiter.join([
        lib_dir,
        os.path.join(lib_dir, 'stanford-parser\stanford-parser.jar'),
        os.path.join(lib_dir, 'stanford-parser\stanford-parser-3.5.1-models.jar'),
        os.path.join(lib_dir, 'easyccg\easyccg.jar')])


def make_dirs(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)


def build_vocab_from_token_files(filepaths, lower=True):
    vocab = set()
    print('Building vocabulary from token files...')
    for filepath in tqdm(filepaths):
        with open(filepath) as f:
            for line in f:
                if lower:
                    line = line.lower()
                vocab |= set(line.split())
    return vocab


def build_vocab_from_items(items, lower=True):
    vocab = set()
    print('Building vocabulary from items...')
    for item in tqdm(items):
        if lower:
            item = item.lower()
        vocab |= {item}
    return vocab


def save_vocab(destination, vocab):
    print('Writing vocabulary: ' + destination)
    with open(destination, 'w', encoding='utf-8') as f:
        for w in tqdm(sorted(vocab)):
            f.write(w + '\n')


def move_numbers_from_known(vocab, vocab_unk):
    r = re.compile(r"(\+|\-)?\d+(\.\d+)?(\/\d+)?")
    to_move = []
    for v in vocab:
        if r.match(v) is not None:
            to_move.append(v)
    vocab -= set(to_move)
    vocab_unk |= set(to_move)
    return vocab, vocab_unk


def tokenize(filepath):
    print('Tokenizing ' + filepath)
    dirpath = os.path.dirname(filepath)
    filepre = os.path.splitext(os.path.basename(filepath))[0]
    tokpath = os.path.join(dirpath, filepre + '.tokens')
    cmd = ('java -cp %s Tokenize -tokpath %s < %s'
           % (classpath, tokpath, filepath))
    os.system(cmd)


def parse_for_variables(filepath, vocab_unk):
    print('Parsing variables ' + filepath)
    dirpath = os.path.dirname(filepath)
    filepre = os.path.splitext(os.path.basename(filepath))[0]
    varpath = os.path.join(dirpath, filepre + '.variables')
    with open(filepath, 'r') as datafile, \
         open(varpath, 'w') as vfile:
        for line in datafile:
            tokens = set(line.split(" "))
            variables = tokens & vocab_unk
            ln = " ".join(variables) + "\n"
            vfile.write(ln)


def dependency_parse(filepath, vocabpath):
    print('Dependency parsing ' + filepath)
    dirpath = os.path.dirname(filepath)
    filepre = os.path.splitext(os.path.basename(filepath))[0]
    parentpath = os.path.join(dirpath, filepre + '.dependency_parents')
    relpath = os.path.join(dirpath, filepre + '.dependency_rels')
    cmd = ('java -cp %s DependencyParse -parentpath %s -relpath %s < %s'
        % (classpath, parentpath, relpath, filepath))
    os.system(cmd)


def constituency_parse(filepath, vocabpath):
    print('Constituency parsing ' + filepath)
    dirpath = os.path.dirname(filepath)
    filepre = os.path.splitext(os.path.basename(filepath))[0]
    parentpath = os.path.join(dirpath, filepre + '.constituency_parents')
    cmd = ('java -cp %s ConstituencyParse -parentpath %s < %s'
        % (classpath, parentpath, filepath))
    os.system(cmd)


def ccg_parse(filepath, vocabpath):
    print('CCG parsing ' + filepath)
    dirpath = os.path.dirname(filepath)
    filepre = os.path.splitext(os.path.basename(filepath))[0]
    parentpath = os.path.join(dirpath, filepre + '.ccg_parents')
    cmd = ('java -cp %s CCGParse -parentpath %s -modelpath lib/easyccg/model < %s'
           % (classpath, parentpath, filepath))
    os.system(cmd)


def parse(filepath, vocabpath):
    dependency_parse(filepath, vocabpath)
    constituency_parse(filepath, vocabpath)
    ccg_parse(filepath, vocabpath)


# loading GLOVE word vectors
# if .pth file is found, will load that
# else will load from .txt file & save
def load_word_vectors(path):
    if os.path.isfile(path+'.pth') and os.path.isfile(path+'.vocab'):
        print('Glove file found, loading to memory...')
        vectors = torch.load(path+'.pth')
        vocab = Vocab(filename=path+'.vocab')
        return vocab, vectors
    # saved file not found, read from txt file
    # and create tensors for word vectors
    print('Glove file not found, preparing, be patient...')
    count = sum(1 for line in open(path+'.txt'))
    with open(path+'.txt', 'r') as f:
        contents = f.readline().rstrip('\n').split(' ')
        dim = len(contents[1:])
    words = [None]*(count)
    vectors = torch.zeros(count,dim)
    with open(path+'.txt','r') as f:
        idx = 0
        for line in f:
            contents = line.rstrip('\n').split(' ')
            words[idx] = contents[0]
            vectors[idx] = torch.Tensor(list(map(float, contents[1:])))
            idx += 1
    with open(path+'.vocab', 'w') as f:
        for word in words:
            f.write(word+'\n')
    vocab = Vocab(filename=path+'.vocab')
    torch.save(vectors, path+'.pth')
    return vocab, vectors