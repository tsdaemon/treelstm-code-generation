import re
import os
import logging
import ast
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import astor

from lang.parse import tokenize_code


def tokenize_for_bleu_eval(code):
    code = re.sub(r'([^A-Za-z0-9_])', r' \1 ', code)
    code = re.sub(r'([a-z])([A-Z])', r'\1 \2', code)
    code = re.sub(r'\s+', ' ', code)
    code = code.replace('"', '`')
    code = code.replace('\'', '`')
    tokens = [t for t in code.split(' ') if t]

    return tokens


def evaluate_decode_result(data,
                           result_id,
                           decode_candidats,
                           out_dir,
                           data_type):

    enc_tree, query, ref_code_tree, ref_code = data

    f = open(os.path.join(out_dir, 'exact_match.txt'), 'w')
    exact_match_ids = []
    f_decode = open(os.path.join(out_dir, 'decode_results.txt'), 'w')

    # eid_to_annot = dict()
    # if config.data_type == 'django':
    #     for raw_id, line in enumerate(open(DJANGO_ANNOT_FILE)):
    #         eid_to_annot[raw_id] = line.strip()

    f_bleu_eval_ref = open(os.path.join(out_dir, 'ref.txt'), 'w')
    f_bleu_eval_hyp = open(os.path.join(out_dir, 'hyp.txt'), 'w')
    f_generated_code = open(os.path.join(out_dir, 'geneated_code.txt'), 'w')

    oracle_bleu = 0.0
    oracle_acc = 0.0
    acc = 0.0
    sm = SmoothingFunction()

    all_references = []
    all_predictions = []

    if len(decode_candidats) == 0:
        logging.debug('Empty decoding results for id {}.'.format(result_id))
        return 0.0, 0.0, 0.0, 0.0, [], []

    ref_ast_tree = ast.parse(ref_code).body[0]
    refer_source = astor.to_source(ref_ast_tree).strip()
    refer_tokens = tokenize_code(refer_source)

    decode_cand = decode_candidats[0]
    cid, cand, ast_tree, code = decode_cand
    code = astor.to_source(ast_tree).strip()

    try:
        predict_tokens = tokenize_code(code)
    except:
        logging.error('error in tokenizing [%s]', code)

    if refer_tokens == predict_tokens:
        acc += 1

        exact_match_ids.append(result_id)
        f.write('-' * 60 + '\n')
        f.write('example_id: {}\n'.format(result_id))
        f.write(code + '\n')
        f.write('-' * 60 + '\n')

    # if data_type == 'django':
    #     ref_code_for_bleu = example.meta_data['raw_code']
    #     pred_code_for_bleu = de_canonicalize_code(code, example.meta_data['raw_code'])
    #     # ref_code_for_bleu = de_canonicalize_code(ref_code_for_bleu, example.meta_data['raw_code'])
    #     # convert canonicalized code to raw code
    #     for literal, place_holder in example.meta_data['str_map'].iteritems():
    #         pred_code_for_bleu = pred_code_for_bleu.replace('\'' + place_holder + '\'', literal)
    #         # ref_code_for_bleu = ref_code_for_bleu.replace('\'' + place_holder + '\'', literal)
    # elif config.data_type == 'hs':
    ref_code_for_bleu = ref_code
    pred_code_for_bleu = code

    # we apply Ling Wang's trick when evaluating BLEU scores
    refer_tokens_for_bleu = tokenize_for_bleu_eval(ref_code_for_bleu)
    pred_tokens_for_bleu = tokenize_for_bleu_eval(pred_code_for_bleu)

    # The if-chunk below is for debugging purpose, sometimes the reference cannot match with the prediction
    # because of inconsistent quotes (e.g., single quotes in reference, double quotes in prediction).
    # However most of these cases are solved by cannonicalizing the reference code using astor (parse the reference
    # into AST, and regenerate the code. Use this regenerated one as the reference)
    weired = False
    if refer_tokens_for_bleu == pred_tokens_for_bleu and refer_tokens != predict_tokens:
        # cum_acc += 1
        weired = True
    elif refer_tokens == predict_tokens:
        # weired!
        # weired = True
        pass

    shorter = len(pred_tokens_for_bleu) < len(refer_tokens_for_bleu)

    all_references.append([refer_tokens_for_bleu])
    all_predictions.append(pred_tokens_for_bleu)

    ngram_weights = [0.25] * min(4, len(refer_tokens_for_bleu))
    bleu = sentence_bleu([refer_tokens_for_bleu], pred_tokens_for_bleu, weights=ngram_weights, smoothing_function=sm.method3)

    logging.info('raw_id: {}, bleu_score: {}'.format(result_id, bleu))

    f_decode.write('-' * 60 + '\n')
    f_decode.write('example_id: %d\n' % result_id)
    f_decode.write('intent: \n')

    # if config.data_type == 'django':
    #     f_decode.write(eid_to_annot[example.raw_id] + '\n')
    # elif config.data_type == 'hs':
    f_decode.write(' '.join(query) + '\n')
    f_bleu_eval_ref.write(' '.join(refer_tokens_for_bleu) + '\n')
    f_bleu_eval_hyp.write(' '.join(pred_tokens_for_bleu) + '\n')
    f_decode.write('canonicalized reference: \n')
    f_decode.write(refer_source + '\n')
    f_decode.write('canonicalized prediction: \n')
    f_decode.write(code + '\n')
    f_decode.write('reference code for bleu calculation: \n')
    f_decode.write(ref_code_for_bleu + '\n')
    f_decode.write('predicted code for bleu calculation: \n')
    f_decode.write(pred_code_for_bleu + '\n')
    f_decode.write('pred_shorter_than_ref: %s\n' % shorter)
    f_decode.write('weired: %s\n' % weired)
    f_decode.write('-' * 60 + '\n')

    # for Hiro's evaluation
    f_generated_code.write(pred_code_for_bleu.replace('\n', '#NEWLINE#') + '\n')

    # compute oracle
    best_score = 0.
    cur_oracle_acc = 0.
    for decode_cand in decode_candidats:
        cid, cand, ast_tree, code = decode_cand
        try:
            code = astor.to_source(ast_tree).strip()
            predict_tokens = tokenize_code(code)

            if predict_tokens == refer_tokens:
                cur_oracle_acc = 1

            # if config.data_type == 'django':
            #     pred_code_for_bleu = de_canonicalize_code(code, example.meta_data['raw_code'])
            #     # convert canonicalized code to raw code
            #     for literal, place_holder in example.meta_data['str_map'].iteritems():
            #         pred_code_for_bleu = pred_code_for_bleu.replace('\'' + place_holder + '\'', literal)
            # elif config.data_type == 'hs':
            pred_code_for_bleu = code

            # we apply Ling Wang's trick when evaluating BLEU scores
            pred_tokens_for_bleu = tokenize_for_bleu_eval(pred_code_for_bleu)

            ngram_weights = [0.25] * min(4, len(refer_tokens_for_bleu))
            bleu_score = sentence_bleu([refer_tokens_for_bleu], pred_tokens_for_bleu,
                                       weights=ngram_weights,
                                       smoothing_function=sm.method3)

            if bleu_score > best_score:
                best_score = bleu_score

        except:
            continue

    oracle_bleu += best_score
    oracle_acc += cur_oracle_acc

    f.close()
    f_decode.close()
    f_bleu_eval_ref.close()
    f_bleu_eval_hyp.close()
    f_generated_code.close()

    return bleu, oracle_bleu, acc, oracle_acc, refer_tokens_for_bleu, pred_tokens_for_bleu

