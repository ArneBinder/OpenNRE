import subprocess

import nrekit
import os
import json

from model_demo import model, get_name


def main(dataset: ('name of dataset in data folder', 'option', 'd', str)='nyt',
         encoder: ('encoder', 'option', 'e', str)='pcnn',
         selector: ('selector', 'option', 's', str)='att',
         add_embeddings: ('add these type of embeddings if not None', 'option', 'a', str)=None,
         reprocess: ('reprocess data', 'flag', 'r') = False
         ):
    model.encoder = encoder
    model.selector = selector
    model_name = get_name(dataset, model.encoder, model.selector, add_embeddings)
    out_fn = os.path.join('./test_result', model_name + "_pred.json")
    dataset_dir = os.path.join('./data', dataset)
    if add_embeddings is not None:
        print('use embeddings: %s' % add_embeddings)
    else:
        print('do not use prepared embeddings')
    if reprocess:
        print('ATTENTION: reprocess data')
    if os.path.exists(out_fn) and not reprocess:
        print('result file already exists: %s' % out_fn)
        pred_result = json.load(open(out_fn))
    else:
        if not os.path.isdir(dataset_dir):
            raise Exception("[ERROR] Dataset dir %s doesn't exist!" % (dataset_dir))

        # The first 3 parameters are train / test data file name, word embedding file name and relation-id mapping file name respectively.
        train_loader = nrekit.data_loader.json_file_data_loader(os.path.join(dataset_dir, 'train.json'),
                                                                os.path.join(dataset_dir, 'word_vec.json'),
                                                                os.path.join(dataset_dir, 'rel2id.json'),
                                                                mode=nrekit.data_loader.json_file_data_loader.MODE_RELFACT_BAG,
                                                                shuffle=True,
                                                                add_embeddings=add_embeddings,
                                                                reprocess=reprocess)
        test_loader = nrekit.data_loader.json_file_data_loader(os.path.join(dataset_dir, 'test.json'),
                                                               os.path.join(dataset_dir, 'word_vec.json'),
                                                               os.path.join(dataset_dir, 'rel2id.json'),
                                                               mode=nrekit.data_loader.json_file_data_loader.MODE_ENTPAIR_BAG,
                                                               shuffle=False,
                                                               add_embeddings=add_embeddings,
                                                               reprocess=reprocess)

        framework = nrekit.framework.re_framework(train_loader, test_loader)

        auc, pred_result = framework.test(model, ckpt=os.path.join('train', model_name, 'checkpoint', model_name), return_result=True)

        if not os.path.exists('./test_result'):
            os.makedirs('./test_result')
        with open(out_fn, 'w') as outfile:
            json.dump(pred_result, outfile)

    if dataset.lower().startswith('semeval'):
        rel2id_fn = os.path.join('./data', dataset, 'rel2id.json')
        rel2id = json.load(open(rel2id_fn))
        fn_predictions_semeval = os.path.splitext(out_fn)[0] + '_semeval.tsv'

        fn_gold_tsv = os.path.join(dataset_dir, 'TEST_FILE_KEY_DIRECTION.TXT')
        if not os.path.exists(fn_gold_tsv):
            fn_gold_full = os.path.join(dataset_dir, 'TEST_FILE_FULL.TXT')
            print('%s not found. convert %s ...' % (fn_gold_tsv, fn_gold_full))
            convert_semeval_full_test_file_to_key_w_direction(fn_gold_full, fn_gold_tsv)

        eval_as_semeval(pred_result, rel2id, fn_predictions_semeval,
                        fn_gold_tsv=os.path.join(dataset_dir, 'TEST_FILE_KEY_DIRECTION.TXT'),
                        fn_script=os.path.join(dataset_dir, 'semeval2010_task8_scorer-v1.2.pl'),
                        fn_eval=os.path.splitext(out_fn)[0] + '_semeval_eval.txt')


def convert_semeval_full_test_file_to_key_w_direction(fn_test_file_full, fn_test_file_key_direction):
    lines_in = list(open(fn_test_file_full).readlines())
    res = []
    for i in range(0, len(lines_in), 4):
        res.append((lines_in[i].split('\t')[0], lines_in[i + 1].strip()))
    open(fn_test_file_key_direction, 'w').writelines(('%s\t%s\n' % (_id, _rel) for _id, _rel in res))


def eval_as_semeval(res, rel2id, fn_predictions_tsv, fn_gold_tsv, fn_script, fn_eval):
    rel2id_rev = {v: k for k, v in rel2id.items()}
    _res = {}

    for record in res:
        _id = int(record['entpair'].split('/')[0])
        _rel = rel2id_rev[record['relation']]
        if _rel == 'NA':
            _rel = 'Other'
        _score = record['score']
        _res.setdefault(_id, {})[_rel] = _score

    _res_max = {}
    for _id in _res:
        scores = _res[_id]
        _res_max[_id] = max(scores, key=(lambda key: scores[key]))

    print('write predictions formatted for semeval2010task8 to %s' % fn_predictions_tsv)
    with open(fn_predictions_tsv, 'w') as f:
        f.writelines(('%i\t%s\n' % (_id, _res_max[_id]) for _id in sorted(_res_max)))

    check_script = 'perl %s %s %s' % (fn_script, fn_predictions_tsv, fn_gold_tsv)
    perl_result = subprocess.check_output(check_script, shell=True).decode("utf-8")
    open(fn_eval, 'w').write(perl_result)
    last_line = perl_result.split('\n')[-2]
    score_str = last_line.replace('<<< The official score is (9+1)-way evaluation with directionality taken into account: macro-averaged F1 = ', '').replace('% >>>', '')
    f1 = float(score_str) / 100
    print('macro-averaged F1 = %f' % f1)
    return f1


if __name__ == '__main__':
    import plac; plac.call(main)
    print('done')
