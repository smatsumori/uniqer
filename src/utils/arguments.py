import os
import sys
import argparse
import yaml
import copy

import torch


def get_args(args=None):
    """
    Parameters
    ----------
    args: list
        such as ['--batchsize', '200', '--epoch', '100', '-g']
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='IVQG Framework on CLEVR dataset',
    )

    # ==============================
    # Paths
    # ==============================
    # yaml setting file
    parser.add_argument(
        '--yaml_path', type=str, default=None,
        help='A path to .yaml file which stores training parameter settings.'
    )

    # dataset
    parser.add_argument(
        '--question_data_version', type=str, default='ask',
        help='Dataset version management (should not contain "/")'
    )
    parser.add_argument(
        '--image_data_version', type=str, default='CLEVR_v1.0',
        help=('Image dataset version. Should be either '
              'CLEVR_v1.0 or CLEVR_Ask+.')
    )
    parser.add_argument(
        '--multi_cpu', action='store_true',
        help='Multi threading mode in cropped image generation.'
    )

    # load models
    parser.add_argument(
        '--load_dir', type=str, default=None,
        help='load models from specified directory'
    )
    parser.add_argument(
        '--load_generator', type=str, default=None,
        help='path of the lstm generator weight to be loaded.'
    )
    parser.add_argument(
        '--load_state_encoder', type=str, default=None,
        help='path of the lstm state encoder weight to be loaded.'
    )
    parser.add_argument(
        '--load_relational_net', type=str, default=None,
        help='path of the relational networks weight to be loaded.'
    )
    parser.add_argument(
        '--load_qa_encoder', type=str, default=None,
        help='path of the lstm q&a encoder weight to be loaded.'
    )
    parser.add_argument(
        '--load_guesser', type=str, default=None,
        help='path of the guesser weight to be loaded.'
    )
    parser.add_argument(
        '--load_splitter', type=str, default=None,
        help='path of the splitter weight to be loaded.'
    )

    # Pre-calculated Image Features
    parser.add_argument(
        '--load_train_context_feature', type=str, default=None,
        help="path of the context image(training) feature's npz file"
    )
    parser.add_argument(
        '--load_val_context_feature', type=str, default=None,
        help="path of the context image(validation) feature's npz file"
    )
    parser.add_argument(
        '--load_test_context_feature', type=str, default=None,
        help="path of the context image(evaluation) feature's npz file"
    )
    parser.add_argument(
        '--load_train_cropped_feature', type=str, default=None,
        help="path of the cropped object image(training) feature's npz file"
    )
    parser.add_argument(
        '--load_val_cropped_feature', type=str, default=None,
        help="path of the cropped object image(validation) feature's npz file"
    )
    parser.add_argument(
        '--load_test_cropped_feature', type=str, default=None,
        help="path of the cropped object image(evaluation) feature's npz file"
    )
    parser.add_argument(
        '--load_guesser_train_obj_feature', type=str, default=None,
        help=("path of the cropped object image(training) feature's npz file "
              "predicted by Guesser")
    )
    parser.add_argument(
        '--load_guesser_val_obj_feature', type=str, default=None,
        help=("path of the cropped object image(validation) feature's npz file"
              "predicted by Guesser")
    )
    parser.add_argument(
        '--load_guesser_test_obj_feature', type=str, default=None,
        help=("path of the cropped object image(evaluation) feature's npz file"
              "predicted by Guesser")
    )

    # save models
    parser.add_argument(
        '--save_dir', type=str, default=None,
        help='create a directory then save models there.'
    )
    parser.add_argument(
        '--n_restricted_objs', type=int, default=7500,
        help=('The number of restricted objects. This value corresponds to'
              'the number of New Objects.')
    )
    parser.add_argument(
        '--vocab_path', type=str, default='data/ask/savedata/vocab.json',
        help='path of the vocab.json.'
    )
    parser.add_argument(
        '--metadata_path', type=str,
        default='./src/datagen/envfiles/metadata_CLEVR_v1.0.json',
        help='path of the CLEVR dataset metadata.json.'
    )
    parser.add_argument(
        '--scene_train_path', type=str,
        default='scenes/CLEVR_train_scenes.json',
        help='path of the CLEVR train dataset scene.json.'
    )
    parser.add_argument(
        '--scene_val_path', type=str,
        default='scenes/CLEVR_val_scenes.json',
        help='path of the CLEVR val dataset scene.json.'
    )
    parser.add_argument(
        '--scene_test_path', type=str,
        default='scenes/CLEVR_test_scenes.json',
        help='path of the CLEVR evaluation(test) dataset scene.json.'
    )
    parser.add_argument(
        '--train_prefix', type=str,
        default='CLEVR_train_',
        help=''
    )
    parser.add_argument(
        '--val_prefix', type=str,
        default='CLEVR_val_',
        help=''
    )
    parser.add_argument(
        '--test_prefix', type=str,
        default='CLEVR_test_',
        help=''
    )
    parser.add_argument(
        '--gen_zip', type=str, default=None,
        help='generate zip files containing pre-trained models'
    )
    parser.add_argument(
        '--gdown_url', type=str, default=None,
        help='GoogleDrive file url (uid or full url)'
    )
    parser.add_argument(
        '--dialogue_h5_train_path', type=str, default=None,
        help='If given, load virtual dialogue(training) saved in h5 file'
    )
    parser.add_argument(
        '--dialogue_h5_val_path', type=str, default=None,
        help='If given, load virtual dialogue(validation) saved in h5 file'
    )
    parser.add_argument(
        '--dialogue_h5_test_path', type=str, default=None,
        help='If given, load virtual dialogue(evaluation) saved in h5 file'
    )


    # ==============================
    # Hyper Parameters & Modeling
    # ==============================

    # -------
    # RL
    # -------
    parser.add_argument(
        '--rl_lr', type=float, default=0.01,
        help='initial learning rate'
    )
    parser.add_argument(
        '--rl_epochs', type=int, default=5, metavar='N',
        help='the number of epochs (for Reinforcement Learning)'
    )
    # TODO: require use same images for every epoch
    # NOTE: in this case, currently, you should add `--train_not_shuffle`
    parser.add_argument(
        '--n_max_episodes', type=int, default=70000, metavar='N',
        help='the number of max episodes in an epoch (WIP)'
    )
    parser.add_argument(
        '--n_max_questions', type=int, default=8, metavar='N',
        help='the number of max questions'
    )
    parser.add_argument(
        '--n_max_steps', type=int, default=100, metavar='N',
        help='the number of max steps (= dialogue * word_per_question)'
    )
    parser.add_argument(
        '--split_class_size', type=int, default=3, metavar='N',
        help='the number of split classes when splitter mode'
    )
    parser.add_argument(
        '--clip_value', type=float, default=0.05,
        help='gradient clipping value for RL'
    )
    parser.add_argument(
        '--gamma', type=float, default=1.0,
        help='a discount factor'
    )
    parser.add_argument(
        '--turn_discount_coeff', type=float, default=0.2,
        help='turn discount coefficient'
    )
    parser.add_argument(
        '--discounted_corr', action='store_true',
        help='Turn discounted for correct reward.'
    )
    parser.add_argument(
        '--turn_penalty', action='store_true',
        help='turn taking penalty'
    )
    parser.add_argument(
        '--informativeness_reward', action='store_true',
        help='add informativeness rewards'
    )
    parser.add_argument(
        '--progressive_reward', action='store_true',
        help='add progressive rewards'
    )
    parser.add_argument(
        '--optimal_reward', action='store_true',
        help='add optimal rewards'
    )
    parser.add_argument(
        '--descriptive_reward', action='store_true',
        help='add descriptive rewards'
    )
    parser.add_argument(
        '--descriptive_coeff', type=float, default=0.05,
        help='the coefficient for descriptive rewards'
    )
    parser.add_argument(
        '--rl_freeze', action='store_true',
        help='freeze Guesser, StateEncoder and QAEncoder'
    )
    parser.add_argument(
        '--value_estimator', action='store_true',
        help='activate baseline function (activation = reward - baseline)'
    )

    # RL network splits object
    # You can add `--proposed_model_mode` to use stronger Guesser
    parser.add_argument(
        '--split_obj_model_mode', action='store_true',
        help='RL is responsible for the part of dividing objects.'
    )
    parser.add_argument(
        '--otm_type', type=str, default='GRU', choices=['GRU', 'transformer'],
        help='Object Targeting Module architecture type'
    )
    parser.add_argument(
        '--rl_single_tf_mode', action='store_true',
        help='RL with SingleTransFormer Model'
    )

    # proposed Guesser & Generator with top_k obj info  # TODO RENAME!!
    parser.add_argument(
        '--proposed_model_mode', action='store_true',
        help='proposed guesser model will be used'
    )

    # beta Generator
    parser.add_argument(
        '--beta_generator_mode', action='store_true',
        help=('pre-calculated Guesser features will be used '
              '(required proposed_model_mode True currently)')
    )
    # idk_mode
    parser.add_argument(
        '--idk_mode', action='store_true',
        help='I dont know mode for oracle.'
    )
    parser.add_argument(
        '--ans_vocab_size', type=int, default=3,
        help=('Default answers area {0: "No", 1: "Yes", 2: "NAN"}'
              'If idk_mode is true, {3: "IDK"} is added.')
    )

    parser.add_argument(
        '--rl_log_interval_batch_ep_iter', type=int, default=1, metavar='N',
        help='When RL, logging every X batch iterations.'
    )
    parser.add_argument(
        '--rl_save_model_interval_epoch', type=int, default=1, metavar='N',
        help='When RL, save models every X epoch.'
    )

    # force stop mode
    parser.add_argument(
        '--force_stop', action='store_true',
        help='Force to generate <STOP> token at the final question.'
    )

    # -------
    # others
    # -------
    parser.add_argument(
        '--n_max_steps_per_question', type=int, default=10,
        help='The number of steps allowed for question generation.'
    )
    parser.add_argument(
        '--lr', type=float, default=1e-3,
        help='initial learning rate'
    )
    parser.add_argument(
        '--adabelief', action='store_true',
        help='Single tf mode can use AdaBelief instead of Adam Optimizer'
    )
    parser.add_argument(
        '--num_warmup_epochs', type=int, default=0,
        help='Single tf mode can use warmup scheduler if value > 0'
    )
    parser.add_argument(
        '--batch_size', type=int, default=1024, metavar='N',
        help='batch size'
    )
    parser.add_argument(
        '--epochs', type=int, default=300, metavar='N',
        help='max epoch size'
    )
    parser.add_argument(
        '--data_size_ratio', type=float, default=1, metavar='N',
        help='N * 100% of full data used for fast experiment'
    )
    parser.add_argument(
        '--patience', type=int, default=5, metavar='N',
        help=('patience (How many times in a row do you stop learning '
              'if your losses do not decrease)')
    )
    # dummy = [<start> <filiter_color[red]> <scene> <end> <null> ... <null>]
    # `gen_seq_len` should be `len(dummy) - 1` (remove <start>)
    parser.add_argument(
        '--gen_seq_len', type=int, default=7, metavar='N',
        help=('The maximum length of the generating question by Question '
              'Generator. Since the size has already been decided when '
              'pre-processed them, we need to adjust to this. However, '
              '<start> token will not be generated, therefore, you must '
              'set to `MAX_SEQ_LENGTH - 1`.')
    )
    # NOTE: Recommendation 18 for resnet101 (around 80% model)
    parser.add_argument(
        '--bottleneck_id', type=int, default=-1, metavar='N',
        help=('On PyTorch, the sub-blocks of the resnet can be defined as '
              'BottleNeck (for resnet50, resnet101, resnet152). Therefore, '
              'it is possible to extract 1D image features by cutting out '
              'the model with bottleneck. When the id is -1, use second to'
              'the last layer to extract features.')
    )
    # TODO: better to automatically detects ?
    parser.add_argument(
        '--context_img_enc_dim', type=int, default=1024, metavar='N',
        help='dimension of context image features.'
    )
    # TODO: better to automatically detects ?
    parser.add_argument(
        '--cropped_img_enc_dim', type=int, default=256, metavar='N',
        help='dimension of cropped object image features.'
    )
    parser.add_argument(
        '--qa_state_enc_dim', type=int, default=256, metavar='N',
        help='dimension of Questions and Answers encoding. (StateEncoder)'
    )
    parser.add_argument(
        '--qa_hidden_dim', type=int, default=256, metavar='N',
        help='dimension of Questions and Answers encoding. (QA Encoder)'
    )

    # TODO: rename argument
    # NOTE: if proposed_model_mode is True --> Guesser's top-k objects will be
    #       used as initial hidden states in Question Generator
    # NOTE: if split_obj_model_mode is True --> only top-k objects will be
    #       divided into the two clusters
    parser.add_argument(
        '--proposed_q_gen_top_k', type=int, default=5, metavar='N',
        help=("guesser's top-k objects features will be used as inital hidden "
              "state of Question Generator.")
    )
    parser.add_argument(
        '--load_epoch', type=int, default=1, metavar='N',
        help='For rl_test mode, specifying model path with saved epoch'
    )
    parser.add_argument(
        '--guesser_dropout', type=float, default=0.0,
        help="Guesser MLP's dropout ratio."
    )
    parser.add_argument(
        '--resnet_type', type=str, default='resnet101',
        help=('resnet version to be loaded. '
              '(Currently, supporting only presave image vec)')
    )
    parser.add_argument(
        '--train_not_shuffle', action='store_true',
        help=('turn off shuffling of the training data loader to eliminate '
              'the possibility of using a different image for each epoch.')
    )

    # ==============================
    # Scripts
    # ==============================
    parser.add_argument(
        '--train_single_tf', action='store_true',
        help='pre-training single transformer model (proposed model)'
    )
    parser.add_argument(
        '--obj_memory_only', action='store_true',
        help=('Send only the output corresponding to the part of the input '
              'object feature to the decoder as memory')
    )
    parser.add_argument(
        '--check_single_tf', action='store_true',
        help='check sample results of pre-trained single transformer model'
    )
    parser.add_argument(
        '--check_rl', action='store_true',
        help='check if the rl module works'
    )
    parser.add_argument(
        '--train_rl', action='store_true',
        help='train reinforcement learning'
    )
    parser.add_argument(
        '--test_by_top_k_questions', action='store_true',
        help=('test reinforcement learning results on NewImage & NewObject by '
              'just asking top_k generated questions (For A85k Dataset only)')
    )
    parser.add_argument(
        '--gen_guesser_feature_vec', action='store_true',
        help='Pre-saving Guesser obj_feature_vectors using pre-trained Guesser'
    )
    parser.add_argument(
        '--enc_only', action='store_true',
        help='pre-training only Encoder of single transformer model'
    )

    # ==============================
    # Others
    # ==============================
    parser.add_argument(
        '--seed', type=int, default=76,
        help='seed value'
    )
    parser.add_argument(
        '--gpu_id', type=int, default=4,
        help='gpu id (0 ~ max_gpu_num-1)'
    )
    parser.add_argument(
        '--multi_gpu', action='store_true',
        help='flag for using multi_gpu (default: False) (will be deprecated?)'
    )
    parser.add_argument(
        '--num_workers', type=int, default=0,
        help='torch dataloader num_workers (choose carefully!)'
    )
    parser.add_argument(
        '--not_pin_memory', action='store_true',
        help=('By default, we use torch automatic memory pinning, but you can '
              'desactivate by specifying this argument.')
    )
    parser.add_argument(
        '--off_wandb', action='store_true',
        help='No tracking using wandb when the argument is given.'
    )
    parser.add_argument(
        '--wandb_project', type=str, default='default_project',
        help='Specify the name of wandb project.'
    )
    parser.add_argument(
        '--running_reward', type=float, default=0.0, metavar='N',
        help='logging running reward rate (?)'
    )
    parser.add_argument(
        '--running_discount', type=float, default=0.0, metavar='N',
        help='logging running discount rate'
    )
    parser.add_argument(
        '--rl_save_log_interval', type=int, default=100, metavar='N',
        help='When RL, save log every XX episodes'
    )
    parser.add_argument(
        '--n_question_per_image', type=int, default=20, metavar='N',
        help=('Depends on dataset (requiring pre-save the info, then this '
              'argument automatically updated)')
    )
    parser.add_argument(
        '--force_overwrite', action='store_true',
        help=('When true then force overwrite save directories.')
    )
    parser.add_argument(
        '--check_summarizer', action='store_true',
        help='Check summarizer module'
    )
    parser.add_argument(
        '--summary_saver', type=str, default='wkhtmltopdf',
        help=(
            'Selects summary save backend from'
            "['wkhtmltopdf', 'pyppeteer', 'None']."
            'The option wkhtmltopdf requires package installation to'
            'your native environment while pyppeteer does not.'
            'If you select None it will skip summary savings.'
        )
    )

    if args is not None:
        args = parser.parse_args(args=args)
    else:
        args = parser.parse_args()
    return args


def yaml_loader(yml_path, args, logger):
    """
    Loads .yaml file and update args.

    Parameters
    ----------
    yml_path : str
        A path to .yaml.
    args : ArgumentParser
    logger : logger

    Returns
    -------
    args : ArgumentParser
    """
    with open(yml_path, 'r') as rf:
        params_data = yaml.load(rf)

    # update arguments
    for k, v in params_data.items():
        args.__dict__[k] = v

    # HARDCODED
    # replace TAGs such as "<SAVE_DIR>"
    tags = {
        '<SAVE_DIR>': args.save_dir,
        '<IMAGE_DATA_V>': args.image_data_version,
    }
    _args = vars(copy.deepcopy(args))
    for k, v in _args.items():
        if isinstance(v, str):
            for tag, tag_v in tags.items():
                v = v.replace(tag, tag_v)
                args.__dict__[k] = v

    return args


def update_args(args, logger):
    """
    Default arugments update settings are here.

    Parameters
    ----------
    args : ArgumentParser
    logger : logger

    Returns
    -------
    args : ArgumentParser
    """
    # load yaml file then update args
    if args.yaml_path:
        args = yaml_loader(args.yaml_path, args, logger)

    # device check
    if torch.cuda.is_available():
        logger.info(f'Using cuda:{args.gpu_id} for training.')
        if args.gpu_id >= torch.cuda.device_count():
            previous_id = args.gpu_id
            args.gpu_id = torch.cuda.device_count() - 1
            logger.warning(
                f'gpu id updated to {args.gpu_id} from {previous_id}'
            )
    else:
        logger.warning('GPU not available!')

    # save path update
    # FIXME: not elegant
    if (args.save_dir is None) and (
            args.train_guesser or
            args.check_guesser or
            args.train_state_encoder or
            args.check_state_encoder or
            args.train_generator or
            args.check_generator or
            args.train_single_tf or
            args.check_single_tf or
            args.train_rl
    ):
        logger.error('args.save_dir should be specified !')
        sys.exit()
    elif args.save_dir is None:
        args.save_dir = 'tmp'

    # ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---
    # update save model path depends on script type
    # ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---
    save_postfix = ''
    if args.train_rl:
        save_postfix = '_rl'
    # elif args.train_relational_net:
    #     # TODO: impl. if necessary
    #     raise NotImplementedError

    args.save_generator = os.path.join(
        args.save_dir, f'generator{save_postfix}.pt'
    )
    args.save_guesser = os.path.join(
        args.save_dir, f'guesser{save_postfix}.pt'
    )
    args.save_qa_encoder = os.path.join(
        args.save_dir, f'qa_encoder{save_postfix}.pt'
    )
    args.save_state_encoder = os.path.join(
        args.save_dir, f'state_encoder{save_postfix}.pt'
    )
    args.save_relational_net = os.path.join(
        args.save_dir, f'relational_net{save_postfix}.pt'
    )
    args.save_splitter = os.path.join(
        args.save_dir, f'splitter{save_postfix}.pt'
    )

    # ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---
    # update loading model path depends on script type
    # ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---
    if args.check_rl:
        # TODO: refactoring here
        args.load_dir = args.save_dir
    elif args.load_dir is None and not args.train_rl:
        args.load_dir = args.save_dir
        logger.info(f'models would be loaded from {args.load_dir}')
    elif args.load_dir is None and args.train_rl:
        logger.error('You should specify load_dir in train_rl model.')
        sys.exit()
    elif (args.load_dir == args.save_dir) and args.train_rl:
        logger.error(
            'save_dir should be different from load_dir in train_rl mode'
        )
        sys.exit()
    else:
        logger.info('train_rl mode successfully initialized!')
        logger.info(f'Load directory : {args.load_dir}')
        logger.info(f'Save directory : {args.save_dir}')

    if not os.path.isdir(args.load_dir) and args.load_dir != 'tmp':
        logger.warning(f'args.load_dir({args.load_dir}) does not exist!')
        sys.exit()

    load_postfix = ''

    if (args.train_rl) or \
         (args.check_rl):
        # loading `--train_generator`'s pre-trained model
        load_postfix = '_m'
    elif args.test_by_top_k_questions:
        load_postfix = '_rl'
    # elif args.train_relational_net:
    #     # TODO: impl. if necessary
    #     raise NotImplementedError

    # epoch_id is '_N' or ''
    epoch_id = ''
    if args.test_by_top_k_questions:
        epoch_id = '_' + str(args.load_epoch)

        # NOTE: I implemented this to execute train and test with  a single
        #       yaml file, but it can be easily misunderstood...
        args.load_dir = args.save_dir

    load_postfix = load_postfix + epoch_id

    load_generator = os.path.join(args.load_dir, f'generator{load_postfix}.pt')
    if os.path.isfile(load_generator):
        args.load_generator = load_generator

    load_guesser = os.path.join(args.load_dir, f'guesser{load_postfix}.pt')
    if os.path.isfile(load_guesser):
        args.load_guesser = load_guesser

    load_state_encoder = os.path.join(
        args.load_dir, f'state_encoder{load_postfix}.pt'
    )
    if os.path.isfile(load_state_encoder):
        args.load_state_encoder = load_state_encoder

    load_qa_encoder = os.path.join(
        args.load_dir, f'qa_encoder{load_postfix}.pt'
    )
    if os.path.isfile(load_qa_encoder):
        args.load_qa_encoder = load_qa_encoder

    load_relational_net = os.path.join(
        args.load_dir, f'relational_net{load_postfix}.pt'
    )
    if os.path.isfile(load_relational_net):
        args.load_relational_net = load_relational_net

    load_splitter = os.path.join(args.load_dir, f'splitter{save_postfix}.pt')
    if os.path.isfile(load_splitter):
        args.load_splitter = load_splitter

    # update vocab_path based on dataset version (not elegant)
    args.vocab_path = args.vocab_path.replace(
        'ask', args.question_data_version
    )

    if args.idk_mode:
        args.ans_vocab_size = 4
        logger.info('Running in idk_mode. Answer vocab size is 4.')

    if args.split_obj_model_mode:
        args.stop_ids = [
            0, args.split_class_size ** args.proposed_q_gen_top_k - 1
        ]
        logger.info(f'Split class size: {args.split_class_size}')
        logger.info(f'Class <{args.stop_ids}> will be regarded as <STOP>.')
    return args


def save_info(args, logger):
    """
    Supposed to be called from main.py

    Parameters
    ----------
    args : ArgumentParser
    logger : logger
    """
    logger.info('=' * 80)
    logger.info(' ' * 5 + 'Arguments' + ' ' * 17 + '|' + ' ' * 5 + 'Values')
    logger.info('-' * 80)
    results_args = sorted(vars(args).items())
    for arg, value in results_args:
        logger.info(f' {arg:30}: {value}')
    logger.info('=' * 80)
