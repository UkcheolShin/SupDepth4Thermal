# Written by Ukcheol Shin (shinwc159[at]gmail.com)
import dataloaders.custom_transforms as cs_tf

def get_augmentations(modal, imgsize, flags):
    # augmentation for input images in training stage
    tf_aug = list()
    if flags.FlagHFlip:  tf_aug.append(cs_tf.RandomHorizontalFlip())
    if flags.FlagCrop:   tf_aug.append(cs_tf.RandomScaleCenterCrop())
    if flags.FlagJitter: tf_aug.append(cs_tf.ColorAugTransform(Itype=modal))

    # augmentation for input images in training + eval stages
    tf_eval = list()
    if flags.FlagResize: tf_eval.append(cs_tf.RescaleTo(imgsize, flags['resize_depth']))
    tf_eval.append(cs_tf.ArrayToTensor(Itype=modal))
    tf_eval.append(cs_tf.TensorImgEnhance(modal, flags))
    tf_eval.append(cs_tf.Normalize())

    return tf_aug+tf_eval, tf_eval

def get_dataloaders(name, tf_dict, opt, split='train_val', eval_mode=None):
    if name == 'MS2':
        from dataloaders.loaders.MS2_dataset import DataLoader_MS2 as DataLoader
    elif (name == 'ViViD') or ('ViViD' in name):
        from dataloaders.loaders.ViViD_dataset import DataLoader_ViViD as DataLoader
    else:
        raise ValueError('Unknown dataset type: {}.'.format(name))

    dataset = {}
    if 'train' in split:
        dataset['train']        =   DataLoader(
                                    opt.dataset_dir,
                                    tf_dict=tf_dict,
                                    data_split = 'train',                                  
                                    data_format=opt.data_format,
                                    modality=opt.train.modality,
                                    sampling_step=opt.train.sample_step,
                                    set_length=opt.train.seq_length,
                                    set_interval=opt.train.seq_interval,
                                    opt=opt
                                )
    if 'val' in split:
        dataset['val']={}
        if 'depth' in eval_mode:
            dataset['val']['depth'] =   DataLoader(
                                        opt.dataset_dir,
                                        tf_dict=tf_dict,
                                        data_split = 'val',                                  
                                        data_format=opt.data_format,
                                        modality=opt.val.modality,
                                        sampling_step=opt.val.sample_step,
                                        set_length=opt.val.seq_length,
                                        set_interval=opt.val.seq_interval,
                                        opt=opt
                                    )
    if 'test' in split:
        dataset['test']={}
        if 'depth' in eval_mode:
            dataset['test']['depth'] =  DataLoader(
                                        opt.dataset_dir,
                                        tf_dict=tf_dict,
                                        data_split = opt.test_env \
                                        if 'test_env' in opt.keys() else 'test',                                  
                                        data_format=opt.data_format,
                                        modality=opt.test.modality,
                                        sampling_step=opt.test.sample_step,
                                        set_length=opt.test.seq_length,
                                        set_interval=opt.test.seq_interval,
                                        opt=opt
                                    )

    return dataset

def get_dataset(opt_dataset, opt_augmentation, split='train_val', eval_mode=None):
    # Set train/eval data augmentator for avaiable modalities (e.g., RGB, NIR, THR images)
    tf_dict = {}
    for modal in opt_dataset['available_modality']:
        tf_dict[modal] = {}
        if split == 'train_val':
            img_size = opt_dataset[modal]['train_size']
            opt_augmentation[modal].resize_depth = True
        else:
            img_size = opt_dataset[modal]['test_size']
            opt_augmentation[modal].resize_depth = False

        augs_ = get_augmentations(modal=modal, imgsize=img_size, \
                                  flags=opt_augmentation[modal])
        tf_dict[modal]['train']  = cs_tf.CustomCompose(augs_[0])
        tf_dict[modal]['eval']   = cs_tf.CustomCompose(augs_[1])

    tf_dict['do_nothing']  = cs_tf.CustomCompose([cs_tf.do_nothing()])

    # Set dataloader 
    dataset={}
    name = opt_dataset['name']
    dataset = get_dataloaders(name, tf_dict, opt_dataset,\
                              split=split, eval_mode=eval_mode)
    return dataset

def build_dataset(opt_datasets, eval_mode, split='train_val'):
    """
    Return corresponding dataset with given dataset option
    :param opt_dataset
    :return dataset
    """
    dataset_list = opt_datasets['list']
    opt_aug = opt_datasets['Augmentation']

    # first dataset is the main dataset
    opt_main_dataset = opt_datasets[dataset_list[0]]
    dataset = get_dataset(opt_main_dataset, opt_aug,\
                          split=split, eval_mode=eval_mode)

    # other datasets are used additional dataset for training sets
    if len(dataset_list) > 1:
        # crawling train sets
        train_sets = [dataset['train']]
        for name in dataset_list[1:]:
            dataset_sub = get_dataset(opt_datasets[name], opt_aug, split='train')
            train_sets.append(dataset_sub['train'])

        # build unified dataloader
        from dataloaders.loaders.multiple_dataset import MultipleDataLoader
        opt = opt_datasets['multi']
        dataset['train'] = MultipleDataLoader(train_sets, opt)

    return dataset
