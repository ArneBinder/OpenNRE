import nrekit
import os

from model_demo import model, get_name


def main(dataset='nyt', encoder='pcnn', selector='att', use_prepared_embeddings=False):
    dataset_dir = os.path.join('./data', dataset)
    if not os.path.isdir(dataset_dir):
        raise Exception("[ERROR] Dataset dir %s doesn't exist!" % (dataset_dir))

    if use_prepared_embeddings:
        print('use prepared embeddings')
    else:
        print('do not use prepared embeddings')
    # The first 3 parameters are train / test data file name, word embedding file name and relation-id mapping file name respectively.
    train_loader = nrekit.data_loader.json_file_data_loader(os.path.join(dataset_dir, 'train.json'),
                                                            os.path.join(dataset_dir, 'word_vec.json'),
                                                            os.path.join(dataset_dir, 'rel2id.json'),
                                                            mode=nrekit.data_loader.json_file_data_loader.MODE_RELFACT_BAG,
                                                            shuffle=True,
                                                            use_prepared_embeddings=use_prepared_embeddings)
    test_loader = nrekit.data_loader.json_file_data_loader(os.path.join(dataset_dir, 'test.json'),
                                                           os.path.join(dataset_dir, 'word_vec.json'),
                                                           os.path.join(dataset_dir, 'rel2id.json'),
                                                           mode=nrekit.data_loader.json_file_data_loader.MODE_ENTPAIR_BAG,
                                                           shuffle=False,
                                                           use_prepared_embeddings=use_prepared_embeddings)

    framework = nrekit.framework.re_framework(train_loader, test_loader)

    model.encoder = encoder
    model.selector = selector

    #model_name = dataset + "_" + model.encoder + "_" + model.selector + '_pe' + str(use_prepared_embeddings)
    model_name = get_name(dataset, model.encoder, model.selector, use_prepared_embeddings)
    dir_train = os.path.join('./train', model_name)
    os.makedirs(dir_train)
    dir_checkpoint = os.path.join(dir_train, "checkpoint")
    dir_summary = os.path.join(dir_train, "summary")

    framework.train(
        model, ckpt_dir=dir_checkpoint,
        summary_dir=dir_summary,
        model_name=model_name,
        max_epoch=60, gpu_nums=1)


if __name__ == '__main__':
    import plac; plac.call(main)
