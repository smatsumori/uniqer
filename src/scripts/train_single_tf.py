import os
from shutil import copyfile

from tqdm import tqdm
from statistics import mean
from fastprogress import master_bar, progress_bar
import torch
import wandb
from adabelief_pytorch import AdaBelief

from modules.single_tf import SingleNet
from modules.oracle.oracle import Oracle
from utils.utils import load_vocab, seed_everything
from utils.dataloader import get_dataloader, get_test_loader
from utils.consts import RICH_DIM, BASE_DIM
from utils.scheduler import get_linear_schedule_with_warmup

# logging
from logging import getLogger
logger = getLogger(__name__)


def train_single_tf(args):
    """
    Parameters
    ----------
    args : ArgumentParser
    """
    seed_everything(args.seed)
    device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Detected device type: {device}')

    # Copy the yaml to the temporal directory
    dest = os.path.join(args.save_dir, os.path.basename(args.yaml_path))
    if args.yaml_path != dest:
        copyfile(args.yaml_path, dest)

    # wandb
    if not args.off_wandb:
        wandb.init(
            project='single_model',
            entity='aaai_2020',
            name=os.path.basename(args.save_dir),
        )
        wandb.config.update(args)
        wandb.save(dest)

    vocab = load_vocab(args.vocab_path)

    # Modeling
    spatial_dim = RICH_DIM if args.proposed_model_mode else BASE_DIM
    net = SingleNet(
        device,
        args.cropped_img_enc_dim + spatial_dim,
        vocab,
        args.ans_vocab_size,
        args.gen_seq_len,
        args.load_guesser,
        args.load_generator,
        args.proposed_q_gen_top_k,
        args.split_class_size,
        args.image_data_version,
        max_batch_size=args.batch_size,
        obj_memory_only=args.obj_memory_only,
    )

    # Data Loading
    train_loader, val_loader = get_dataloader(args, multi_q_ver=True)

    # ========================================================================
    # Configurations
    # ========================================================================
    # optimizer
    if args.enc_only:
        logger.info('Training only Encoder(=Guesser) Mode !')
        params = net.encoder.parameters()
    else:
        params = list(net.encoder.parameters()) + list(net.decoder.parameters())

    if args.adabelief:
        # ref: https://github.com/juntang-zhuang/Adabelief-Optimizer
        optimizer = AdaBelief(
            params,
            args.lr,
            eps=1e-16,
            betas=(0.9, 0.999),
            weight_decouple=True,
            rectify=False,  # use SGD for warmup at first when True
        )
    else:
        optimizer = torch.optim.Adam(params, args.lr)

    # scheduler
    if args.num_warmup_epochs > 0:
        logger.info('LinearScheduleWarmup using!')
        data_size = len(train_loader.dataset)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            args.num_warmup_epochs * (data_size // args.batch_size),
            args.epochs * (data_size // args.batch_size),
        )

    init_epoch = 1
    best_loss = 1e7
    early_stop_count = 0

    # TODO: load pre-trained model & info

    # Learning
    mb = master_bar(range(init_epoch, args.epochs + 1))
    for epoch in mb:
        train_so_losses = []
        train_prg_losses = []
        valid_so_losses = []
        valid_prg_losses = []

        # Training loop
        for data in progress_bar(train_loader, parent=mb):
            # TODO: track guesser loss & qgen loss separately
            train_so_loss, train_prg_loss = net.pretrain_net(
                data,
                is_train=True,
                enc_only=args.enc_only,
                proposed_model_mode=args.proposed_model_mode,
                idk_mode=args.idk_mode,
            )
            optimizer.zero_grad()
            if args.enc_only:
                train_so_loss.backward()
            else:
                (train_so_loss + train_prg_loss).backward()

            optimizer.step()
            if args.num_warmup_epochs > 0:
                # update scheduler
                scheduler.step()

            train_so_losses.append(train_so_loss.item())
            train_prg_losses.append(train_prg_loss.item())

        # Validation loop
        for data in progress_bar(val_loader, parent=mb):
            valid_so_loss, valid_prg_loss = net.pretrain_net(
                data,
                is_train=False,
                enc_only=args.enc_only,
                proposed_model_mode=args.proposed_model_mode,
                idk_mode=args.idk_mode,
            )
            valid_so_losses.append(valid_so_loss.item())
            valid_prg_losses.append(valid_prg_loss.item())

        # --------------------
        # Model Saving
        # --------------------
        epoch_loss = mean(valid_so_losses) + mean(valid_prg_losses)
        if epoch_loss < best_loss:
            # update model
            best_loss = epoch_loss
            early_stop_count = 0
            save_dict = {
                'next_epoch': epoch + 1,
                'best_loss': best_loss,
                'optimizer': optimizer.state_dict(),
            }
            net.save_models(save_dict, args.save_guesser, args.save_generator)
        else:
            early_stop_count += 1
            if early_stop_count == args.patience:
                logger.info(f'Early Stopping: epoch {epoch}')
                break

        # logging
        msg = (
            'epoch: {}/{} - lr: {:.6f} - '
            'train_so_loss: {:.5f} - val_so_loss: {:.5f} - '
            'train_prg_loss: {:.5f} - val_prg_loss: {:.5f} - '
            'train_loss: {:.5f} - val_loss: {:.5f}'
        ).format(
            epoch,
            args.epochs,
            optimizer.state_dict()['param_groups'][0]['lr'],
            mean(train_so_losses),
            mean(valid_so_losses),
            mean(train_prg_losses),
            mean(valid_prg_losses),
            mean(train_so_losses) + mean(train_prg_losses),
            mean(valid_so_losses) + mean(valid_prg_losses),
        )
        logger.info(msg)

        # ----------------------
        # Logging wandb
        # ----------------------
        if not args.off_wandb:
            curr_lr = optimizer.state_dict()['param_groups'][0]['lr']
            logging_info_dict = {
                'learning_rate': curr_lr,
                'train_so_loss': mean(train_so_losses),
                'valid_so_loss': mean(valid_so_losses),
                'train_prg_loss': mean(train_prg_losses),
                'valid_prg_loss': mean(valid_prg_losses),
                'train_loss': mean(train_so_losses) + mean(train_prg_losses),
                'valid_loss': mean(valid_so_losses) + mean(valid_prg_losses),
                'best_loss': best_loss,
            }
            wandb.log(logging_info_dict)


def check_single_tf(args):
    """
    Parameters
    ----------
    args : ArgumentParser
    """
    seed_everything(args.seed)
    device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Detected device type: {device}')

    vocab = load_vocab(args.vocab_path)

    # Modeling
    spatial_dim = RICH_DIM if args.proposed_model_mode else BASE_DIM
    net = SingleNet(
        device,
        args.cropped_img_enc_dim + spatial_dim,
        vocab,
        args.ans_vocab_size,
        args.gen_seq_len,
        args.load_guesser,
        args.load_generator,
        args.proposed_q_gen_top_k,
        args.split_class_size,
        args.image_data_version,
        max_batch_size=args.batch_size,
        obj_memory_only=args.obj_memory_only,
    )

    oracle = Oracle(
        args.metadata_path,
        os.path.join('data/', args.image_data_version, args.scene_test_path),
        args.vocab_path,
        args.image_data_version,
    )

    # Data Loading
    test_loader = get_test_loader(args, multi_q_ver=True)

    for i_data, data in enumerate(tqdm(test_loader)):
        show_example = True if i_data == 0 else False
        net.check_results(
            data,
            oracle,
            args.proposed_model_mode,
            args.idk_mode,
            show_example=show_example
        )
    net.show_total_results()