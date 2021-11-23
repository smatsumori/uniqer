from fastprogress import master_bar, progress_bar
import numpy as np
import torch

from modules.rl.environment_sp import EnvironmentSp

from modules.rl.policy import Policy
from modules.rl.agent import Agent
from scripts.train_rl import summarize_ep_values, calculate_epoch_values

from utils.utils import seed_everything, load_vocab
from utils.dataloader import get_rl_dataloader, get_rl_test_loader

from utils.rl.rl_batch_looper_sp import BatchLooperSp

from utils.rl.rl_utils import TrainLogger
from utils.rl.register_metrics import init_metrics


# Logging
from logging import getLogger  # noqa: E402
logger = getLogger(__name__)


def train_rl(
    args, agent=None,
    train_env=None, valid_env=None, test_o_env=None, test_i_env=None,
):
    # TODO: merge into `train_rl.py - train_rl()`
    # Initialize seeds
    seed_everything(args.seed)
    device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'

    # Initialize train logger
    # If you want to track custom metrics, update the init_metrics function.
    metrics = init_metrics(args)

    # TODO: We might need to change the logging steps
    tlogger = TrainLogger(
        args,
        project=args.wandb_project,
        entity='aaai_2020',
        logger=logger,
        track=metrics,
    )

    vocab = load_vocab(args.vocab_path)
    vocab_size = len(vocab['program_token_to_idx'])

    # Instantiate the environment
    if not train_env:  # for train data
        train_env = EnvironmentSp(args)
    if not valid_env:  # for val (new image) data
        valid_env = EnvironmentSp(args, is_valid=True)
    if not test_o_env:  # new object
        test_o_env = EnvironmentSp(args, is_eval_obj=True)
    if not test_i_env:  # new image
        test_i_env = EnvironmentSp(args, is_eval_img=True)

    # Instantiate the agent
    if not agent:
        # Build the policy network
        policy = Policy(args, device, vocab_size)
        agent = Agent(policy, split_mode=args.split_obj_model_mode)

    # Baseline value estimate function (takes hidden states of qgen)
    # TODO: impl.
    if args.value_estimator and args.split_obj_model_mode:
        raise NotImplementedError

    valueest = None

    # Dataloader
    train_loader, valid_loader = get_rl_dataloader(args)
    test_o_loader = train_loader  # New object mode uses train dataset
    test_i_loader = get_rl_test_loader(args)

    cumieps = 0  # cumulative i_episodes
    mb = master_bar(range(1, args.rl_epochs + 1))
    for epoch in mb:
        # 1. Train Loop
        epoch_loop(
            'train', args, train_env, agent, valueest, train_loader,
            cumieps, mb, epoch, device, tlogger
        )
        # 2. Validation Loop
        epoch_loop(
            'valid', args, valid_env, agent, valueest, valid_loader,
            cumieps, mb, epoch, device, tlogger
        )
        # 3. Test Loop
        # 3-1. New Object Evaluation
        epoch_loop(
            'test_o', args, test_o_env, agent, valueest, test_o_loader,
            cumieps, mb, epoch, device, tlogger
        )
        # 3-2. New Image Evaluation
        epoch_loop(
            'test_i', args, test_i_env, agent, valueest, test_i_loader,
            cumieps, mb, epoch, device, tlogger
        )

        # reset oracle's refenrece object
        train_env.reinit_reference_object()
        # prehaps unnecessary
        valid_env.reinit_reference_object()

        # NOTE: (No rest on test set)
        # test_i_env.reinit_reference_object()
        # test_o_env.reinit_reference_object()

    logger.info('Done!')


def epoch_loop(
    mode, args, env, agent, valueest, data_loader,
    cumieps, mb, epoch, device, tlogger
):
    # Check mode is valid
    print()  # for visualization
    logger.info(f'>>>>> Mode: {mode} <<<<<')
    if mode == 'train':
        summary_prefix = ''
        is_train = True
        enum_iter = enumerate(progress_bar(data_loader, parent=mb))
        image_base_path = f'./data/{args.image_data_version}/images/train/'
        n_samples_to_save = 0
        n_samples_to_upload = 0
        n_samples_to_track = 0
    elif mode == 'valid':
        summary_prefix = 'val_'
        is_train = False
        enum_iter = enumerate(progress_bar(data_loader, parent=mb))
        image_base_path = f'./data/{args.image_data_version}/images/val/'
        dialogue_summaries = []
        n_samples_to_save = 0
        n_samples_to_upload = 0
        n_samples_to_track = 0
    elif mode in ['test_i', 'test_o']:
        summary_prefix = mode + '_'
        is_train = False
        enum_iter = enumerate(progress_bar(data_loader, parent=mb))
        dset = 'train' if mode == 'test_o' else 'test'
        image_base_path = f'./data/{args.image_data_version}/images/{dset}/'
        dialogue_summaries = []
        n_samples_to_save = 30
        n_samples_to_upload = 30
        n_samples_to_track = 5 if mode == 'test_i' else 0
    else:
        assert False

    ieps = 0  # reset index of episodes

    eps_values_list = []  # placeholder
    epoch_values = None  # placeholder

    # Start episodes (loop for unique images in dataset)
    looper = BatchLooperSp(
        args.n_max_questions, env.tokens, args.stop_ids,
        args.force_stop,
    )

    # TODO: handle case where mb is none (valid, test)
    for ib, data in enum_iter:
        # Fetch new scenes. Target object is random.
        state, batch_size = env.next_scene(data)
        env.init_batches(batch_size)

        # batch_size = data['n_obj'].shape[0]
        looper.init_batches(n_batches=batch_size)  # initializes looper
        ieps += batch_size
        cumieps += batch_size

        # TODO: move this to dataloader
        if is_train and ieps > args.n_max_episodes:
            break

        # --------------- Begin define metrics ---------------
        # current episode rewards
        ep_reward = 0.0
        ep_corr_r = 0.0
        ep_info_r = 0.0
        ep_prog_r = 0.0
        ep_opti_r = 0.0
        ep_turn_p = 0.0
        ep_disc_p = 0.0
        ep_desc_r = 0.0

        # losses
        ep_valueest_loss = 0
        ep_policy_loss = 0

        # other metrics
        ep_n_step = 0
        ep_variety = 0
        ep_tk_variety = 0
        ep_invalid_ratio = 0

        # helper metrics
        # TODO: implement invalid and related metrics
        invalid_question_count = [0 for _ in range(batch_size)]
        related_queston_count = [0 for _ in range(batch_size)]
        # --------------- End define metrics ---------------

        # Question generation steps.
        for q in range(1, args.n_max_questions + 1):
            # 0. Model initialization
            # Guesser Init. Generator Init. StateEncoder Init.
            if q == 1:
                # o_pred_prob.shape: (n_batches, nmax_objects)
                o_pred_probs = agent.model_initialize(
                    state, is_train=is_train
                )
                # ve_states = []

            # 1. Generate split order actions (RL)
            # actions.shape: (n_batches)
            actions = agent.act(state, is_train=is_train)
            looper.set_actions(actions)

            # 2. Generate questions (QGen)
            # Split order -> Question tokens
            # questions.shape: (n_batches, n_max_q_length, n_tokens)
            # DEBUG: actions.shape: (n_batches, n_max_obj)
            questions, split_actions = agent.generate_question(state, actions)
            looper.set_questions(questions)

            # 3. Gets answer from Oracle (EnvironmentSp)
            # env_questions: including <Null> token
            state = env.update(
                looper.get_env_actions(),
                split_actions,
                looper.get_env_questions(),
                looper.get_is_currq_eod(),
                is_last_question=looper.is_last_question
            )

            # 4. Update predictions
            o_pred_probs = agent.eoq_update(state)
            looper.set_predictions(o_pred_probs)

            # rewards.shape: (n_batches)
            rws = env.get_batch_reward(
                looper.predictions, looper.question_counts, q,
                looper.get_is_currq_eoq(), looper.get_is_currq_eod()
            )
            (  # TODO: update desc_r
                reward, corr_r, info_r, prog_r, opti_r,
                turn_p, disc_p, desc_r
            ) = rws  # unzip
            # TODO: implement value estimator
            if args.value_estimator:
                raise NotImplementedError
            else:
                advantage = 0
                _eps_rewards = corr_r + disc_p
                _im_rewards = info_r + prog_r + opti_r + turn_p + desc_r

            # Update returns. _eps_reward.shape: (n_batches)
            agent.policy.register_rewards(_eps_rewards, _im_rewards)

            # reset question and answer status for the current dialogue.
            env.reset_batch_questions()

            # Summarize epoch metrics for logging
            ep_reward += np.sum(reward) / batch_size
            ep_corr_r += np.sum(corr_r) / batch_size
            ep_info_r += np.sum(info_r) / batch_size
            ep_prog_r += np.sum(prog_r) / batch_size
            ep_opti_r += np.sum(opti_r) / batch_size
            ep_turn_p += np.sum(turn_p) / batch_size
            ep_disc_p += np.sum(disc_p) / batch_size
            ep_desc_r += np.sum(desc_r) / batch_size

            # if all of batch terminated with eod
            if looper.is_all_batch_eod:
                break

            # Update question generation steps
            looper.qstep()
        # ---- End of the batch episodes -----

        # compute metrics (variety, n_steps)
        # this is not a single episode step number, but summed by batch
        ep_n_step = looper.get_n_all_steps() / batch_size
        # TODO: move to looper?
        vs = [
            env.get_question_variety(bid) for bid in range(batch_size)
        ]
        ep_variety = sum(vs) / batch_size
        ep_tk_variety = env.get_token_variety()

        # Update params
        # 1. policy function
        # TODO: rename ep -> eps
        # loss_mask.shape: (n_batches, n_max_questions)
        ep_policy_loss = agent.policy.update_params(
            loss_mask=looper.loss_masks, no_grad=not is_train
        )

        # 2. baseline function (optional)
        if args.value_estimator:
            # update baseline function
            # vloss = valueest.update_params()
            # ep_valueest_loss += vloss / looper.n_mask_activated
            raise NotImplementedError

        # update logger
        # TODO: move to tlogger
        # compute invalid question ratio
        n_ques = sum(looper.question_counts) / batch_size
        n_invr = sum(invalid_question_count) / batch_size
        n_rel = sum(related_queston_count) / batch_size
        ep_invalid_ratio = (n_invr / n_ques) if n_ques != 0 else 0
        ep_related_ratio = (n_rel / n_ques) if n_ques != 0 else 0
        ep_stop_ratio = sum(looper.eod_status) / batch_size

        # summarize episode metrics
        eps_values = summarize_ep_values(
            ep_reward, ep_corr_r, ep_info_r,
            ep_prog_r, ep_opti_r, ep_turn_p, ep_disc_p, ep_desc_r,
            advantage, ep_n_step, ep_invalid_ratio, ep_related_ratio,
            ep_stop_ratio, ep_variety, ep_tk_variety, n_ques,
            ep_valueest_loss, ep_policy_loss,
            prefix=summary_prefix  # train (''), valid ('val'), test ('test')
        )

        # Store episode rewards for epoch summary
        eps_values_list.append(eps_values)

        if is_train:
            # Update episode metrics (running values are calculated as well)
            # We only track episode metrics for training phase
            tlogger.update_episode_metrics(eps_values)
            # Logging
            if ib % args.rl_log_interval_batch_ep_iter == 0:
                print()  # for visualization
                # logging
                tlogger.print_episode_summary(ieps)
                if not args.off_wandb:
                    tlogger.wandb_log()
        else:
            # TODO: implementation for train?
            dialogue_summaries.extend(env.summarize_current_episodes())
        # ---- End of a batch ----

    # Summarize epoch_values
    # TODO: perhaps we should add epoch summary of training as well
    if not is_train:
        epoch_values = calculate_epoch_values(eps_values_list, ib+1)
        tlogger.update_epoch_metrics(epoch_values)

        # Log epoch results
        tlogger.print_epoch_summary(epoch, flush=True, split=mode)
        if not args.off_wandb:
            tlogger.wandb_log_epoch(epoch, split=mode)

        # Save dialogue summaries
        tlogger.save_dialogue_summary(
            dialogue_summaries, args.save_dir, image_base_path,
            summary_prefix, mode, epoch, save_on_wandb=not args.off_wandb,
            n_samples_to_save=n_samples_to_save,
            n_samples_to_upload=n_samples_to_upload,
            n_samples_to_track=n_samples_to_track
        )

    # Save models
    # TODO: save the model only when it archives the current best result
    # Save models in validation mode
    if mode == 'valid' and epoch % args.rl_save_model_interval_epoch == 0:
        # check is_current_best with tlogger and save the model if so
        if tlogger.is_current_best('val_corr_r', desc=False):
            logger.info('Achieved the current best results. Saving models...')
            # save agent models
            save_dict = tlogger.get_save_dict(cumieps)
            agent.save_models(args, save_dict, epoch, overwrite=True)


def test(args):
    # TEMP
    logger.info('Checking trained model on RL...')
    seed_everything(args.seed)
    device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'

    metrics = init_metrics(args)

    args.off_wandb = True  # to make sure it won't try to connect wandb

    tlogger = TrainLogger(
        args,
        project=args.wandb_project,
        entity='tmp',
        logger=logger,
        track=metrics,
    )

    vocab = load_vocab(args.vocab_path)
    vocab_size = len(vocab['program_token_to_idx'])

    # Instantiate the environment
    test_o_env = EnvironmentSp(args, is_eval_obj=True)
    test_i_env = EnvironmentSp(args, is_eval_img=True)

    # Instantiate the agent
    # Build the policy network
    policy = Policy(args, device, vocab_size)
    agent = Agent(policy, split_mode=args.split_obj_model_mode)
    valueest = None
    mb = master_bar(range(1, 2))

    # Dataloader
    train_loader, _ = get_rl_dataloader(args)
    test_o_loader = train_loader  # New object mode uses train dataset
    test_i_loader = get_rl_test_loader(args)

    cumieps = 0  # cumulative i_episodes
    # dummy loop
    for epoch in mb:
        # 1. New Object Evaluation
        epoch_loop(
            'test_o', args, test_o_env, agent, valueest, test_o_loader,
            cumieps, mb, epoch, device, tlogger
        )
        # 2. New Image Evaluation
        epoch_loop(
            'test_i', args, test_i_env, agent, valueest, test_i_loader,
            cumieps, mb, epoch, device, tlogger
        )

    logger.info('Done!')
