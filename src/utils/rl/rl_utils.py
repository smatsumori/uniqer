import os
import json
from shutil import copyfile

import numpy as np
from PIL import Image
import wandb

from utils.rl.summarizer import summarize_dialogue

# logging
from logging import getLogger
logger = getLogger(__name__)


class MetricNotFound(Exception):
    pass


class TrainLogger():
    """
    Custom logger for reinforcement learning.
    Currently available features are:
    - custom metrics tracking
    - wandb submission
    - print log with logger
    """

    def __init__(self, args, project, entity, logger, track=None):
        """
        Initializes TrainLogger.
        Parameters
        ----------
        args : arguments
        project : a project name used for wandb
        entity : an entity name used for wandb
        logger : a logger
        track : a list of CustomMetrics instances.
            The metrics in the list are automatically tracked.
            If you want to add some metrics after,
            call `register` function.
        """
        # initialization
        self.epoch = 1  # epoch starts from 1
        self.log_step = 1  # log epoch starts from 1
        self.logger = logger
        self.yaml_path = args.yaml_path
        self.summary_saver = args.summary_saver
        if not args.off_wandb:
            self.__init_wandb(args, project, entity)
        else:
            self.logger.warning(
                'Wandb Desactivated as `--off_wandb` argument is given.'
            )

        # track custom metrics
        self.tracked_metrics = []
        if track:
            for m in track:
                self.register(m)

    def __init_wandb(self, args, project, entity):
        """
        Initializes wandb.
        """
        wandb.init(
            project=project,
            entity=entity,
            name=os.path.basename(args.save_dir),
        )
        wandb.config.update(args)
        # Copy the yaml to the temporal directory
        dest = os.path.join(args.save_dir, os.path.basename(self.yaml_path))
        if args.yaml_path != dest:
            copyfile(self.yaml_path, dest)
        # save .yaml path
        wandb.save(dest)

    def save_dialogue_summary(
        self, summaries, save_dir, image_base_path, summary_prefix, mode,
        epoch, save_on_wandb=False, n_samples_to_save=100,
        n_samples_to_upload=100, n_samples_to_track=100,
    ):
        """save_dialogue_summary.

        Parameters
        ----------
        summaries :
            summaries
        save_dir : str
            save_dir
        summary_prefix :
            summary_prefix
        mode : str
            mode in ['train', 'valid', 'test_i', 'test_o']
        epoch : int
            epoch
        save_on_wandb :
            Whether to save artifacts to wandb.
        n_samples_to_save : int
            The number of samples to save to the local folder.
        n_samples_to_upload : int
            The number of samples to upload to wandb.
        n_samples_to_track : int
            The number of samples to track with wandb logger.
            These samples are registered to wandb media board.
        """
        # 1. Save dialogue summaries in json format
        _save_path = os.path.join(save_dir, f'dialogue_json_{mode}')
        os.makedirs(_save_path, exist_ok=True)
        save_path_full = os.path.join(
            _save_path,
            f'dialogue_{summary_prefix}epoch-{epoch}.json'
        )
        with open(save_path_full, 'w') as f:
            json.dump(summaries, f)
        # Upload fraction of summaries for faster execution
        if save_on_wandb:
            _save_path = os.path.join(save_dir, f'dialogue_json_small_{mode}')
            os.makedirs(_save_path, exist_ok=True)
            save_path_small = os.path.join(
                _save_path,
                f'dialogue_{summary_prefix}epoch-{epoch}_small.json'
            )
            with open(save_path_small, 'w') as f:
                json.dump(summaries[:n_samples_to_upload], f)
            wandb.save(save_path_small)

        # 2. Save dialogue summaries in pdf and image format
        if not self.summary_saver == 'None':
            all_artifact_path, indv_artifact_paths = summarize_dialogue(
                summaries, mode, epoch, image_base_path, save_dir,
                limit_save_idxs=[i for i in range(n_samples_to_save)],
                summary_saver=self.summary_saver
            )
            if save_on_wandb and (0 < n_samples_to_save):
                # Upload artifacts (summary.pdf) to wandb.
                wandb.save(all_artifact_path)
                # Load saved images
                images = {}
                for pt in indv_artifact_paths[:n_samples_to_track]:
                    im = Image.open(pt)
                    images[
                        f'{os.path.basename(pt)}'
                    ] = wandb.Image(np.array(im))
                # Upload images to wandb
                wandb.log(images, step=self.log_step)

    def register(self, metric):
        """
        Registers custom metrics.

        Parameters
        ----------
        metric : an instance of CustomMetrics.
        """
        assert isinstance(metric, CustomMetrics)
        self.tracked_metrics.append(metric)
        self.logger.debug(
            'Tracking metrics: {}'.format(metric.name)
        )

    def get_tracked_metrics(self, label):
        """
        Gets a tracked metric instance.

        Parameters
        ----------
        label : str
            This must be the label designated in
            CustomMetrics.
        """
        for m in self.tracked_metrics:
            if m.label == label:
                return m
        # if the metric not found
        raise MetricNotFound

    def update_episode_metrics(self, d):
        """ Updates the episode metrics.

        Parameters
        ----------
        d : dict. Dictionary with labels and values
            of the metrics to be updated.
            {
                '<label_of_custom_metric>': <value_to_be_updated>
                ...
            }
        """
        for k, v in d.items():
            try:
                m = self.get_tracked_metrics(k)
                m.update(v)
            except MetricNotFound:
                self.logger.warning('Metric {} not found!'.format(k))

    def update_epoch_metrics(self, d):
        """ Updates the epoch metrics.
        Parameters
        ----------
        d : dict. Dictionary with labels and values
            of the metrics to be updated.
            {
                '<label_of_custom_metric>': <value_to_be_updated>
                ...
            }
        """
        for k, v in d.items():
            try:
                m = self.get_tracked_metrics(k)
                # Only update the epoch metric
                if m.met_epoch:
                    m.update(v)
            except MetricNotFound:
                self.logger.warning('Metric {} not found!'.format(k))

        # Update the best value when the metric
        # is tracking the other metric
        for k, v in d.items():
            try:
                m = self.get_tracked_metrics(k)
                if m.met_epoch and m.is_tracking:
                    mt = self.get_tracked_metrics(m.track_met)
                    assert mt.met_epoch
                    m.update_best_value(v, mt.is_best)
            except MetricNotFound:
                continue
            except ValueError:
                # TODO: FIXME: check_rl mode
                continue

    def print_episode_summary(self, iep):
        """ Prints episode summaries.

        Parameters
        ----------
        iep : int. an episode iteration count
        """
        avglog = []
        for met in self.tracked_metrics:
            # Skip if the metrics is not for print or
            # if the metric is EPOCH metrics
            if (not met.print_log) or (met.met_epoch):
                continue
            avglog.append(
                '\t[Avg]{}: {:.3f}'.format(met.name, met.running_value)
            )
        self.logger.info(
            f'Episode {iep}' + ''.join(avglog)  # averaged metrics
        )

    def print_epoch_summary(self, iepoch, flush=False, split=None):
        """ Prints epoch summaries.

        Parameters
        ----------
        iepoch : int. an epoch iteration count
        split : str
            Split of metrics. ['train', 'valid', 'test_o', 'test_i']
        """
        if flush:
            print()
        avglog = []
        for met in self.tracked_metrics:
            # Skip if the metrics is not for print or
            # if the metric is EPISODE metrics
            if (not met.print_log) or (not met.met_epoch):
                continue
            # If specified, only print metrics with such split
            if split and split != met.split:
                continue
            try:
                avglog.append(
                    # Fetch the latest value instead of running_value
                    # for the epoch metrics
                    '\t[Avg]{}: {:.3f}'.format(met.name, met.latest)
                )
            except IndexError:
                self.logger.error(
                    'Epoch metric {} has no entry!'.format(met.label)
                )
                continue
        self.logger.info(
            f'Epoch {iepoch}' + ''.join(avglog)  # averaged metrics
        )

    def __get_log_info_dict(self):
        """ Gets log_info_dict.
        """
        # get average metrics
        avgmet = {
            'average_' + m.label: m.running_value
            for m in self.tracked_metrics
            if m.wandb_track and not m.met_epoch
            # only returns episode metrics
        }
        return avgmet

    def __get_epoch_log_info_dict(self, split):
        """ Gets epoch_log_info_dict.
        """
        epoch_met = {}
        # TODO: we shoud make use of episode metrics
        # rather than calculating it in hte main loop
        for m in self.tracked_metrics:
            if m.wandb_track and m.met_epoch:
                if split and split != m.split:
                    continue
                lb = f'epoch_average_{m.label}'
                epoch_met[lb] = m.latest

                # register the best values
                if m.is_tracking:
                    lbb = f'best_{m.label}_tracked_{m.track_met}'
                    epoch_met[lbb] = m.latest_best
        return epoch_met

    def wandb_log(self):
        """ Submits a log to wandb.
        This is called at the end of every training episode.
        """
        logdict = self.__get_log_info_dict()
        logdict['epoch'] = self.epoch
        wandb.log(logdict)
        self.log_step += 1

    def wandb_log_epoch(self, epoch, split=None):
        """ Submits a log to wandb.
        This is called at the end of every epoch.

        Paramters
        ---------
        epoch: int
        split : str
            Split of metrics. ['train', 'valid', 'test_o', 'test_i']
        """
        # Submit to wandb
        self.epoch = epoch
        wandb.log(self.__get_epoch_log_info_dict(split), step=self.log_step)

    def get_save_dict(self, iep):
        """ Gets save_dict. The save_dict will be used to save the model.
            """
        gtm = self.get_tracked_metrics
        save_dict = {
            'last_episode': iep,
            'average_corr_r': gtm('corr_r').running_value,
            'average_info_r': gtm('info_r').running_value,
            'average_prog_r': gtm('prog_r').running_value,
            'running_reward': gtm('reward').running_value,
        }
        return save_dict

    def is_current_best(self, metric_label: str, desc: bool):
        """
        Check if the current value is the best.

        Parameters
        ----------
        metric_label : str
            The label of the metric.
        desc : bool
            The metric becomes better in descending order.
        """
        metric = self.get_tracked_metrics(metric_label)
        return metric.is_current_best(desc)


class CustomMetrics:
    # per-episode, per-step
    def __init__(
        self, label, name, running_ratio=0.8, desc='',
        print_log=True, wandb_track=True, met_epoch=False,
        split='train', track_met=''
    ):
        """
        Initializes CustomMetrics.

        Parameters
        ----------
        label : str. A label used to identify the metric.
            This label is used to specify the metrics on updates.
        name :  str. A name of metrics.
            An user-friendly name will be preferred.
        running_ratio : float. A discount factor used when computing
            a running value.
        desc : str. A description of the metrics.
        print_log : bool. If true, then the metrics will
            be displayed on log print.
        wandb_track : bool. If true, the metrics will be tracked in wandb.
        met_epoch : bool. If true, the metric will be registered as
            an epoch metric and only updated when `update_epoch_metrics`
            is called.
        track_met : str.
            The label of the other metric should be set.
            Note that epoch metric is only allowed for being tracked.
            The value will be updated when the tracked metric has
            marked the best score. Mainly used for the test evaluations.
        """
        self.n_ep = 0
        self.label = label
        self.name = name
        self.desc = desc
        self.ratio = running_ratio
        self.print_log = print_log
        self.wandb_track = wandb_track
        self.met_epoch = met_epoch
        self.is_tracking = False if not track_met else True
        self.track_met = track_met
        assert split in ['train', 'valid', 'test_i', 'test_o']
        self.split = split

        # placeholders
        # The best value will be updated when the
        # tracked metric achieve the best score
        self.best_values = []
        self.values = []
        self.running_values = [0]
        self.time_stamps = []

    def update(self, value):
        """ Updates value on each episode.

        Parameters
        ----------
        value : float.
            The value of the metrics on an episode.
        """
        self.n_ep += 1
        self.values.append(value)
        self.running_values.append(
            self.__compute_running_value(
                self.running_values[-1],
                value
            )
        )

    def update_best_value(self, value, is_best):
        """ Updates the best value.
        This function should be called after all the metrics
        on the current epoch/episode is updated.

        Parameters
        ----------
        value : float.
            The value of the metrics on an episode.

        is_best : bool
            If true then update the `self.best_value`.
        """
        if self.is_tracking:
            if is_best:
                self.best_values.append(value)
            elif not self.best_values:
                self.best_values.apennd(value)

    @property
    def is_best(self):
        """ Checks if the latest value is the best.
        """
        maxv_idx = self.values.index(max(self.values))
        curr_idx = len(self.values) - 1
        return maxv_idx == curr_idx

    @property
    def latest(self):
        """ Returns the latest value.
        """
        try:
            rval = self.values[-1]
        except IndexError:
            logger.error(f'{self.label} has no entry!')
            rval = 0.0
        return rval

    @property
    def latest_best(self):
        """ Returns the latest best value.
        """
        try:
            rval = self.best_values[-1]
        except IndexError:
            logger.error(f'{self.label} has no entry!')
            rval = 0.0
        return rval

    @property
    def running_value(self):
        """ Returns the running value.
        """
        return self.running_values[-1]

    def is_current_best(self, desc=False):
        """ Returns the current best value.

        Parameters
        ----------
        desc : bool
            True for the metric that is better
            for smaller values.
        """
        try:
            if desc:
                prev_min = min(self.values[:-1])
                return self.latest < prev_min
            else:
                prev_max = max(self.values[:-1])
                return prev_max < self.latest
        except ValueError:
            # If no previous value exists return True
            return True

    def __compute_running_value(self, base, new):
        """ Computes the running value.
        """
        return base * self.ratio + new * (1 - self.ratio)
