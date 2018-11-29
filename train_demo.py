import nrekit
import os
import shutil

from model_demo import model, get_name


def main(dataset: ('name of dataset in data folder', 'option', 'd', str)='nyt',
         encoder: ('encoder', 'option', 'e', str)='pcnn',
         selector: ('selector', 'option', 's', str)='att',
         add_embeddings: ('add these type of embeddings if not None', 'option', 'a', str)=None,
         reprocess: ('reprocess data', 'flag', 'r')=False,
         ):
    dataset_dir = os.path.join('./data', dataset)
    if not os.path.isdir(dataset_dir):
        raise Exception("[ERROR] Dataset dir %s doesn't exist!" % (dataset_dir))

    model.encoder = encoder
    model.selector = selector

    # model_name = dataset + "_" + model.encoder + "_" + model.selector + '_pe' + str(use_prepared_embeddings)
    model_name = get_name(dataset, model.encoder, model.selector, add_embeddings)
    dir_train = os.path.join('./train', model_name)
    if not os.path.exists(dir_train):
        os.makedirs(dir_train)
    dir_checkpoint = os.path.join(dir_train, "checkpoint")
    dir_summary = os.path.join(dir_train, "summary")

    if add_embeddings is not None:
        print('use embeddings: %s' % add_embeddings)
    else:
        print('do not use prepared embeddings')
    if reprocess:
        print('ATTENTION: reprocess data and retrain')
        if os.path.exists(dir_checkpoint):
            #os.remove(dir_checkpoint)
            shutil.rmtree(dir_checkpoint)
        if os.path.exists(dir_checkpoint):
            #os.remove(dir_summary)
            shutil.rmtree(dir_summary)

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

    framework.train(
        model, ckpt_dir=dir_checkpoint,
        summary_dir=dir_summary,
        model_name=model_name,
        max_epoch=60, gpu_nums=1)


if __name__ == '__main__':
    import plac; plac.call(main)
