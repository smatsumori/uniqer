import os
import sys
import logging
import subprocess

from utils import log_init, arguments, utils
from scripts.train_single_tf import train_single_tf, check_single_tf
from scripts.train_rl_sp import train_rl as train_rl_sp
from scripts.train_rl_sp import test as test_rl_sp
from utils.rl.summarizer import test_summarize_dialogue


if __name__ == '__main__':
    # logger
    os.makedirs('results/text_logs', exist_ok=True)
    logger = log_init.logger
    logger.info('script start!')

    # arguments
    args = arguments.get_args()

    # update arguments
    args = arguments.update_args(args, logger)
    arguments.save_info(args, logger)

    # seed fix
    utils.seed_everything(args.seed)

    # GPU config
    if not args.multi_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    # Check
    if args.load_train_context_feature is None or \
       args.load_val_context_feature is None or \
       args.load_train_cropped_feature is None or \
       args.load_val_cropped_feature is None:
        logger.error(
            'Paths to the npz files should be given because raw image ver has '
            'not been implemented yet.'
        )

    if args.train_single_tf:
        train_single_tf(args)
    
    elif args.check_single_tf:
        check_single_tf(args)

    elif args.train_rl:
        if args.split_obj_model_mode:
            train_rl_sp(args)

    elif args.check_rl:
        if args.split_obj_model_mode:
            test_rl_sp(args)
        else:
            raise NotImplementedError

    elif args.check_summarizer:
        test_summarize_dialogue()

    elif args.test_by_top_k_questions:
        # test_by_top_k_questions(args)
        raise NotImplementedError

    else:
        logger.warning('You should specify one script such as `--train_rl`.')

    # send message to slack (FIXME will be deprecated before publish repo.)
    if os.path.isfile('scripts/slack_notificate.sh'):
        if args.save_dir is not None:
            msg = args.save_dir
        else:
            msg = 'Running '  # what we should send ?
        command = f'echo {msg} Done ! | scripts/slack_notificate.sh'
        subprocess.call(command, shell=True)
