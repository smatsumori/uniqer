import os
import sys
import random
import pickle
import h5py

import PIL
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from utils.consts import NUM_MAX_OBJ
from utils.consts import DEFAULT_H, DEFAULT_W, \
    BBX_BASE_COEF, XSMALL_COEF, SMALL_COEF, MEDIUM_COEF

from functools import lru_cache

from . import utils
# import utils  # for debug mode

# logging
import logging
logger = logging.getLogger(__name__)
# Desactivate PIL internal logger (which shows unuseful Resize info)
logging.getLogger(PIL.__name__).setLevel(logging.WARNING)


# ------------------------------
# Helper Functions
# ------------------------------
def load_image_fnames(root_dir, ext='.png'):
    """
    Parameters
    ----------
    root_dir : str
    ext : str, default is 'png'

    Returns
    -------
    out : list of str
        list of filenames found in root_dir
    """
    fnames = sorted(os.listdir(root_dir))
    # extract only png files
    for f in fnames[:]:  # x[:] makes a copy of x
        if not f.endswith(ext):
            fnames.remove(f)
    return fnames


def load_image_features(image_feature_path):
    """
    Parameters
    ----------
    image_feature_path : str or None
        If the argument is not None, load image features instead of image
        It must be npz file.

    Returns
    -------
    feature_vecs : np.array or None
    iname2idx : dict or None
        image_filename to index
    """
    if image_feature_path is None:
        return None, None

    elif image_feature_path[-4:] != '.npz':
        logger.error(f'{image_feature_path} should be npz file.')
        sys.exit()

    elif os.path.exists(image_feature_path):
        logger.info(f'Loading {image_feature_path}...')

        # loading image features
        feature_vecs = np.load(image_feature_path)['arr_0']

        # loading dict {image_filename --> idx}
        with open(image_feature_path.replace('.npz', '_k2i.pkl'), 'rb') as f:
            iname2idx = pickle.load(f)

        return feature_vecs, iname2idx
    else:
        logger.error(f'{image_feature_path} does not exist.')
        sys.exit()


def load_spatial_obj_info(scenes=None, feature_path=None):
    """
    Loading each object spatial info: (x, y, w, h)
        `scenes` should be given
    or
    Loading each object spatial info: dims=131
    --> see presave_spatial_vecs.py
        `feature_path` should be given

    Parameters
    ----------
    scenes : dict or None
    feature_path : str or None

    Returns
    -------
    spatial_features : dict
    """
    logger.info('loading object spatial info...')

    if feature_path is None:
        spatial = []

        for scene in scenes['scenes']:
            for obj in scene['objects']:
                center = obj['pixel_coords']
                bbx_coef = BBX_BASE_COEF / center[2]

                # based on object_size, change bbx_coef
                if obj['size'] == 'xsmall':
                    bbx_coef *= XSMALL_COEF
                if obj['size'] == 'small':
                    bbx_coef *= SMALL_COEF
                elif obj['size'] == 'medium':
                    bbx_coef *= MEDIUM_COEF

                h = int(DEFAULT_H * bbx_coef)
                w = int(DEFAULT_W * bbx_coef)
                spatial.append([center[0], center[1], w, h])

        spatial_features = np.array(spatial)
    else:
        with open(feature_path, 'rb') as f:
            spatial_features = pickle.load(f)
    return spatial_features


# ------------------------------
# Core Classes
# ------------------------------
class RlClevrDataset(Dataset):
    """
    CLEVR Dataset for pytorch (Reinforcement Learning version)
    RL does not require question info.

    TODO docstring

    Attributes
    ----------
    """

    def __init__(
            self,
            proposed_model_mode,
            image_dir,
            scene_path,
            image_feature_path,
            image_prefix,
            obj_dir,
            obj_feature_path,
            data_size_ratio=1.0,
            spatial_feature_path=None,
            guesser_obj_feature_path=None,
            transform=None,
    ):
        """
        Parameters
        ----------
        proposed_model_mode : bool
        image_dir : str
            The image directory path of CLEVR
        scene_path : str
            The scene file path of CLEVR
        image_feature_path : str or None
            If the argument is not None, load image features instead of image
        image_prefix : str,
        obj_dir : str
            The object image directory path of CLEVR
        obj_feature_path : str or None
            If the argument is not None, load object image features
        data_size_ratio : float, default is 1.0
            X * 100 % of full data will be used to training
        spatial_feature_path : str or None, default is None
            If you use the our proposed Guesser, you need to load the spatial
            features calculated beforehand.
        guesser_obj_feature_path : str or None, default is None
        transform : torchvision.transforms, default is None
        """
        super().__init__()
        self.proposed_model_mode = proposed_model_mode
        self.scenes = utils.load_json(scene_path)
        self.image_dir = image_dir
        self.n_images = int(
            len(load_image_fnames(self.image_dir)) * data_size_ratio
        )
        self.image_prefix = image_prefix

        # Try to load image pre-calculated features
        self.image_features, self.iname2idx = load_image_features(
            image_feature_path
        )

        # Try to load cropped-image pre-calculated features
        self.obj_dir = obj_dir
        self.obj_features, self.objname2idx = load_image_features(
            obj_feature_path
        )

        # spatial information
        self.spatial = load_spatial_obj_info(
            scenes=self.scenes, feature_path=spatial_feature_path
        )

        # guesser's obj_feature
        self.g_obj_features, self.g_objname2idx = load_image_features(
            guesser_obj_feature_path
        )

        self.transform = transform

        logger.info(f'The number of images: {self.n_images}')

    def __len__(self):
        return self.n_images

    @lru_cache(maxsize=None)
    def __getitem__(self, idx):
        """
        Parameters
        ----------
        idx : int
            index of images

        Returns
        -------
        data : dict
            image : image itself
            image_feature : pre-calculcated image features
            image_filename : image_filename (not only the basename)
            n_obj : the number of object (will be used to making objects)
            obj_image : NotImplemented
            obj_feature : pre-calculcated image features
            spatial : spatial info of each object (x, y, w, h)
        """
        data = {}

        image_filename = os.path.join(
            self.image_dir, f'{self.image_prefix}{idx:0=6}.png'
        )
        data['image_filename'] = image_filename

        # Loading image or image_features
        if self.image_features is None:
            image = Image.open(image_filename).convert('RGB')
            if self.transform:
                image = self.transform(image)
            data['image'] = image
        else:
            image_feature = self.image_features[self.iname2idx[image_filename]]
            data['image_feature'] = torch.FloatTensor(image_feature)

        # object info
        obj_list = self.scenes['scenes'][idx]['objects']
        n_obj = len(obj_list)
        data['n_obj'] = n_obj

        # image feature  (NUM_MAX_OBJ, Img_feature_dim)
        obj_img_base_name = os.path.join(
            self.obj_dir,
            os.path.basename(image_filename)
        )
        obj_img_start_name = '_00'.join(
            list(os.path.splitext(obj_img_base_name))
        )
        start_idx = self.objname2idx[obj_img_start_name]

        objects = self.obj_features[start_idx:start_idx + n_obj]

        # spatial information  (NUM_MAX_OBJ, 4)
        if self.proposed_model_mode:
            # spatial informations
            #     {<image_filename>: [{
            #             'obj_order': [0, 1, 3, 6, 5, 4, 2, 7, 8, 9],
            #             'spatial_features': [...]
            #         }, {from obj_2}, ... {from obj_10}]
            #     }
            spatial_features = self.spatial[os.path.basename(image_filename)]
            obj_feature_data = np.zeros((
                NUM_MAX_OBJ,
                len(spatial_features[0]['spatial_features']) +
                objects.shape[1] * NUM_MAX_OBJ
            ))

            for i in range(n_obj):
                tmp = -objects.shape[1] * (NUM_MAX_OBJ - n_obj)
                limit = tmp if tmp != 0 else None
                obj_feature_data[i, :limit] = np.concatenate([
                    spatial_features[i]['spatial_features'],
                    objects[spatial_features[i]['obj_order']].reshape(-1),
                ])

            data['obj_features'] = obj_feature_data

            if self.g_obj_features is not None:
                # guesser's pre-calculated object features
                # TODO: we don't need to run guesser if this mode is used in RL
                #       --> no need to load obj_features, etc. --> much faster?
                obj_img_start_name = '_00'.join(
                    list(os.path.splitext(image_filename))
                )

                start_idx = self.g_objname2idx[obj_img_start_name]
                data['g_obj_features'] = self.g_obj_features[
                    start_idx: start_idx + NUM_MAX_OBJ
                ]

            # 1. spatial features in the order corresponding to guesser obj_id
            # 2. cropped_image_features
            # spatial_features[i]['org_order_f']: ((5 + 5 + 2) * NUM_MAX_OBJ)
            # It will be used in our proposed question generator model

            # org_obj_spatial_data : (NUM_MAX_OBJ, spatial_feature_dim)
            org_obj_spatial_data = np.zeros((
                NUM_MAX_OBJ, len(spatial_features[0]['org_order_f'])
            ))
            for i in range(n_obj):
                org_obj_spatial_data[i] = spatial_features[i]['org_order_f']
            data['org_spatial_features'] = org_obj_spatial_data

        else:
            spatial = np.zeros((NUM_MAX_OBJ, 4))
            spatial[:n_obj] = self.spatial[start_idx:start_idx + n_obj]
            data['spatial'] = torch.FloatTensor(spatial)

        pad = np.zeros((NUM_MAX_OBJ - n_obj, objects.shape[1]))
        data['obj_image'] = torch.FloatTensor(
            np.concatenate([objects, pad])
        )

        return data


class MultiQAClevrDataset(Dataset):
    """
    CLEVR Dataset (Dialogue) for pytorch

    TODO: docstring

    Attributes
    ----------
    """

    def __init__(
            self,
            image_dir,
            question_h5_path,
            scene_path,
            image_feature_path,
            image_prefix,
            spatial_feature_path=None,
            obj_dir=None,
            obj_feature_path=None,
            data_size_ratio=1.0,
            guesser_obj_feature_path=None,
            top_k=NUM_MAX_OBJ,
            dialogue_h5_path=None,
            raw_image=False,
            transform=None,
    ):
        """
        Parameters
        ----------
        image_dir : str
            The image directory path of CLEVR
        question_h5_path : str
            h5 file generated by `preprocess_question()`
            question, answer, program are already tokenized
        scene_path : str
            The scene file path of CLEVR
        image_feature_path : str or None
            If the argument is not None, load image features instead of image
        image_prefix : str
        spatial_feature_path : str
            Path to the pkl file generated by `presave_spatial_vecs.py`
        obj_dir : str or None, default is None
            The object image directory path of CLEVR
        obj_feature_path : str or None, default is None
            If the argument is None, not loading obj_features
        data_size_ratio : float, default is 1.0
            X * 100 % of full data will be used to training
        guesser_obj_feature_path : str or None
            Pre-calculated Guesser's object features for Generator-beta model
        top_k : int, default is NUM_MAX_OBJ
            placeholder
        dialogue_h5_path : str or None, default is None
            If the arguemnt is given, programs & obj_answers, and dummy_yn
            will be updated by this file. (Experimental)
        raw_image : bool, default is False
        transform : torchvision.transforms, default is None
        """
        super().__init__()
        self.scenes = utils.load_json(scene_path)

        self._load_questions(question_h5_path)
        self.image_dir = image_dir
        self.image_prefix = image_prefix
        n_images = len(load_image_fnames(self.image_dir))

        self.image_features, self.iname2idx = load_image_features(
            image_feature_path
        )

        self.obj_dir = obj_dir
        self.obj_features, self.objname2idx = load_image_features(
            obj_feature_path
        )

        self.spatial_feature_path = spatial_feature_path
        self.spatial = load_spatial_obj_info(
            scenes=self.scenes, feature_path=self.spatial_feature_path
            )

        n_questions = len(self.questions)
        self.q_per_i = int(n_questions / n_images)
        logger.warning(
            f'Assuming there are {self.q_per_i} questions per image. '
            'Please check carefully if you changed the dataset.'
        )

        # If `guesser_obj_feature_path` is None, return None, None
        self.g_obj_features, self.g_objname2idx = load_image_features(
            guesser_obj_feature_path
        )

        # self.top_k = top_k  # currently, no use of top_k here
        self.raw_image = raw_image  # loading raw image flag
        self.transform = transform

        self.n_images = int(n_images * data_size_ratio)
        logger.info(f'The number of images: {self.n_images}')
        logger.info(
            f'The number of questions: {int(n_questions * data_size_ratio)}'
        )

        self.dialogue_h5_path = dialogue_h5_path
        if self.dialogue_h5_path is not None:
            self._load_questions(self.dialogue_h5_path, dialogue_mode=True)

        # placeholder
        # img_memory : {imagefilename: image}
        self.img_memory = {}

    def _load_questions(self, h5_path, dialogue_mode=False):
        """
        Parameters
        ----------
        h5_path : str
            h5 file generated by `preprocess_question()`
            question, answer, program are already tokenized
        dialogue_mode : bool, default is False
        """
        if os.path.exists(h5_path):
            logger.info('Loading questions...')
            with h5py.File(h5_path, 'r') as f:
                if dialogue_mode:
                    self.programs = np.array(f['programs'])
                    self.answers = np.array(f['answers'])
                    self.dummy_yn = np.array(f['responces'])
                else:
                    self.questions = np.array(f['questions'])
                    self.programs = np.array(f['programs'])
                    self.answers = np.array(f['answers'])
        else:
            logger.error(f'{h5_path} is not correct path.')
            sys.exit()

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        """
        TODO: docstring

        Parameters
        ----------
        idx : int
            index of image (not question index!)

        Returns
        -------
        data : dict
        """
        image_filename = os.path.join(
            self.image_dir, f'{self.image_prefix}{idx:0=6}.png'
        )

        image_feature = self.image_features[self.iname2idx[image_filename]]
        if self.dialogue_h5_path is None:
            data = {
                'image': image_feature,
                'question': torch.LongTensor(self.questions[
                    idx * self.q_per_i: (idx + 1) * self.q_per_i
                ]),
                'answer': torch.LongTensor(self.answers[
                    idx * self.q_per_i: (idx + 1) * self.q_per_i
                ]),
                'program': torch.LongTensor(self.programs[
                    idx * self.q_per_i: (idx + 1) * self.q_per_i
                ]),
                'image_filename': image_filename
            }

        else:
            data = {
                'image': image_feature,
                'answer': torch.LongTensor(self.answers[idx]),
                'program': torch.LongTensor(self.programs[idx]),
                'dummy_yn': torch.LongTensor(self.dummy_yn[idx]),
                'image_filename': image_filename
            }

        # loading raw image
        if self.raw_image:
            try:
                src_image = self.img_memory[image_filename]
            except KeyError:
                src_image = Image.open(image_filename).convert('RGB')
                self.img_memory[image_filename] = src_image

            # transform
            # NOTE: It is obviously faster to apply once a transform on loading
            # the image, but DataAugmentation is not possible.
            if self.transform:
                src_image = self.transform(src_image)

            data['src_image'] = src_image

        # Loading object features and padding to NUM_MAX_OBJ
        if self.obj_features is not None:
            # image feature  (NUM_MAX_OBJ, Img_feature_dim)
            obj_img_base_name = os.path.join(
                self.obj_dir, os.path.basename(image_filename)
            )
            obj_img_start_name = '_00'.join(
                list(os.path.splitext(obj_img_base_name))
            )
            start_idx = self.objname2idx[obj_img_start_name]

            n_obj = len(self.scenes['scenes'][idx]['objects'])
            data['n_obj'] = n_obj

            objects = self.obj_features[start_idx:start_idx + n_obj]

            pad = np.zeros((NUM_MAX_OBJ - n_obj, objects.shape[1]))
            data['obj_image'] = torch.FloatTensor(
                np.concatenate([objects, pad])
            )

            if self.spatial_feature_path is None:
                # spatial information  (NUM_MAX_OBJ, 4)  (x, y, w, h)
                spatial = np.zeros((NUM_MAX_OBJ, 4))
                spatial[:n_obj] = self.spatial[start_idx:start_idx + n_obj]
                data['spatial'] = torch.FloatTensor(spatial)
            else:
                # spatial information
                #     {<image_filename>: [{
                #             'obj_order': [0, 1, 3, 6, 5, 4, 2, 7, 8, 9],
                #             'spatial_features': [...]
                #         }, {from obj_2}, ... {from obj_10}]
                #     }
                spatial_features = self.spatial[
                    os.path.basename(image_filename)
                ]
                obj_feature_data = np.zeros((
                    NUM_MAX_OBJ,
                    len(spatial_features[0]['spatial_features']) +
                    objects.shape[1] * NUM_MAX_OBJ
                ))

                for i in range(n_obj):
                    tmp = -objects.shape[1] * (NUM_MAX_OBJ - n_obj)
                    limit = tmp if tmp != 0 else None
                    obj_feature_data[i, :limit] = np.concatenate([
                        spatial_features[i]['spatial_features'],
                        objects[spatial_features[i]['obj_order']].reshape(-1),
                    ])

                # top RICH_DIM features are spatial features
                data['obj_features'] = obj_feature_data

                # 1. spatial features in the order corresponding to
                #    guesser's obj_id
                # 2. cropped_image_features
                # spatial_features[i]['org_order_f']: ((5 + 5 + 2)*NUM_MAX_OBJ)
                # It will be used in our proposed question generator model

                # org_obj_s_data : (NUM_MAX_OBJ, spatial_feature_dim)
                org_obj_s_data = np.zeros((
                    NUM_MAX_OBJ, len(spatial_features[0]['org_order_f'])
                ))
                for i in range(n_obj):
                    org_obj_s_data[i] = spatial_features[i]['org_order_f']
                data['org_spatial_features'] = org_obj_s_data

        if self.g_obj_features is not None:
            # guesser's pre-calculated object features
            # TODO: we don't need to run guesser if this mode is used in RL
            #       --> no need to load obj_features, etc. --> much faster?
            obj_img_start_name = '_00'.join(
                list(os.path.splitext(image_filename))
            )

            start_idx = self.g_objname2idx[obj_img_start_name]
            data['g_obj_features'] = self.g_obj_features[
                start_idx: start_idx + NUM_MAX_OBJ
            ]
        return data


class ClevrDataset(Dataset):
    """
    CLEVR Dataset for pytorch

    TODO: img_memory
    TODO: docstring

    Attributes
    ----------

    """

    def __init__(
            self,
            image_dir,
            question_h5_path,
            scene_path,
            image_feature_path,
            image_prefix,
            spatial_feature_path=None,
            obj_dir=None,
            obj_feature_path=None,
            data_size_ratio=1.0,
            guesser_obj_feature_path=None,
            top_k=NUM_MAX_OBJ,
            dialogue_h5_path=None,
            raw_image=False,
            transform=None,
    ):
        """
        Parameters
        ----------
        image_dir : str
            The image directory path of CLEVR
        question_h5_path : str
            h5 file generated by `preprocess_question()`
            question, answer, program are already tokenized
        scene_path : str
            The scene file path of CLEVR
        image_feature_path : str or None
            If the argument is not None, load image features instead of image
        image_prefix : str,
        spatial_feature_path : str or None
            Path to the pkl file generated by `presave_spatial_vecs.py`
        obj_dir : str or None, default is None
            The object image directory path of CLEVR
        obj_feature_path : str or None, default is None
            If the argument is None, not loading obj_features
        data_size_ratio : float, default is 1.0
            X * 100 % of full data will be used to training
        guesser_obj_feature_path : str or None
            dummy argument
        top_k : int, default is NUM_MAX_OBJ
            only for split_id top_k mode
        dialogue_h5_path : str or None, default is None
            dummy argument
        raw_image : bool, default is False
        transform : torchvision.transforms, default is None
        """
        super().__init__()
        self.scenes = utils.load_json(scene_path)

        self._load_questions(question_h5_path)
        self.image_dir = image_dir
        self.image_prefix = image_prefix
        self.n_questions = int(len(self.questions) * data_size_ratio)
        self.n_images = int(
            len(load_image_fnames(self.image_dir)) * data_size_ratio
        )
        self.image_features, self.iname2idx = load_image_features(
            image_feature_path
        )

        self.obj_dir = obj_dir
        self.obj_features, self.objname2idx = load_image_features(
            obj_feature_path
        )

        self.spatial_feature_path = spatial_feature_path
        self.spatial = load_spatial_obj_info(
            scenes=self.scenes, feature_path=self.spatial_feature_path
        )

        self.top_k = top_k
        self.raw_image = raw_image
        self.transform = transform

        logger.info(f'The number of images: {self.n_images}')
        logger.info(f'The number of questions: {self.n_questions}')

    def _load_questions(self, question_h5_path):
        """
        Parameters
        ----------
        question_h5_path : str
            h5 file generated by `preprocess_question()`
            question, answer, program are already tokenized
        """
        if os.path.exists(question_h5_path):
            logger.info('Loading questions...')
            with h5py.File(question_h5_path, 'r') as f:
                self.questions = np.array(f['questions'])
                self.programs = np.array(f['programs'])
                self.image_idxs = np.array(f['image_idxs'])
                self.answers = np.array(f['answers'])
        else:
            logger.error(f'{question_h5_path} is not correct path.')
            sys.exit()

    def __len__(self):
        return self.n_questions

    def __getitem__(self, idx):
        """
        TODO: image and image_feature should be explicitly separated

        Parameters
        ----------
        idx : int
            index of question

        Returns
        -------
        data : dict
            TODO: docstring
        """
        image_filename = os.path.join(
            self.image_dir,
            f'{self.image_prefix}{self.image_idxs[idx]:0=6}.png'
        )

        image = self.image_features[self.iname2idx[image_filename]]

        data = {
            'image': image,
            'question': torch.LongTensor(self.questions[idx]),
            'answer': torch.LongTensor(self.answers[idx]),
            'program': torch.LongTensor(self.programs[idx]),
            'image_filename': image_filename
        }

        # loading raw_image
        if self.raw_image:
            src_image = Image.open(image_filename).convert('RGB')
            if self.transform:
                src_image = self.transform(src_image)
            data['src_image'] = src_image

        # Loading object features and padding to NUM_MAX_OBJ
        if self.obj_features is not None:
            # image feature  (NUM_MAX_OBJ, Img_feature_dim)
            obj_img_base_name = os.path.join(
                self.obj_dir, os.path.basename(image_filename)
            )
            obj_img_start_name = '_00'.join(
                list(os.path.splitext(obj_img_base_name))
            )
            start_idx = self.objname2idx[obj_img_start_name]

            n_obj = len(self.scenes['scenes'][self.image_idxs[idx]]['objects'])
            data['n_obj'] = n_obj

            objects = self.obj_features[start_idx:start_idx + n_obj]

            pad = np.zeros((NUM_MAX_OBJ - n_obj, objects.shape[1]))
            data['obj_image'] = torch.FloatTensor(
                np.concatenate([objects, pad])
            )

            if self.spatial_feature_path is None:
                # spatial information  (NUM_MAX_OBJ, 4)  (x, y, w, h)
                spatial = np.zeros((NUM_MAX_OBJ, 4))
                spatial[:n_obj] = self.spatial[start_idx:start_idx + n_obj]
                data['spatial'] = torch.FloatTensor(spatial)
            else:
                # spatial information
                #     {<image_filename>: [{
                #             'obj_order': [0, 1, 3, 6, 5, 4, 2, 7, 8, 9],
                #             'spatial_features': [...]
                #         }, {from obj_2}, ... {from obj_10}]
                #     }
                spatial_features = self.spatial[
                    os.path.basename(image_filename)
                ]
                obj_feature_data = np.zeros((
                    NUM_MAX_OBJ,
                    len(spatial_features[0]['spatial_features']) +
                    objects.shape[1] * NUM_MAX_OBJ
                ))

                for i in range(n_obj):
                    tmp = -objects.shape[1] * (NUM_MAX_OBJ - n_obj)
                    limit = tmp if tmp != 0 else None
                    obj_feature_data[i, :limit] = np.concatenate([
                        spatial_features[i]['spatial_features'],
                        objects[spatial_features[i]['obj_order']].reshape(-1),
                    ])

                # top RICH_DIM features are spatial features
                data['obj_features'] = obj_feature_data

                # 1. spatial features in the order corresponding to
                #    guesser's obj_id
                # 2. cropped_image_features
                # spatial_features[i]['org_order_f']: ((5 + 5 + 2)*NUM_MAX_OBJ)
                # It will be used in our proposed question generator model

                # org_obj_s_data : (NUM_MAX_OBJ, spatial_feature_dim)
                org_obj_s_data = np.zeros((
                    NUM_MAX_OBJ, len(spatial_features[0]['org_order_f'])
                ))
                for i in range(n_obj):
                    org_obj_s_data[i] = spatial_features[i]['org_order_f']
                data['org_spatial_features'] = org_obj_s_data

            # -----------------------------------------------------------------
            # split_idx for pre-train
            # randomly assign cluster_id = 1 or cluster_id = 2
            # 'randomly' means cluster_id does not represent 'Yes' obj
            # nor 'No' obj
            random_assign = random.randint(0, 1)
            ans = data['answer'].clone()
            if random_assign == 0:
                # [random_assign : 0] 'Yes' obj will be cluster_id = 1
                # obj_answer2: Yes --> 1, NO --> 2
                splits = 2 - ans
            else:
                # [random_assign : 1] 'Yes' obj will be cluster_id = 2
                # obj_answer3: Yes --> 2, No --> 1
                splits = 1 + ans
            splits[n_obj:] = 0  # TODO: consts 0, 1 and 2

            # top_k
            n_add_mask = n_obj - self.top_k
            if n_add_mask > 0:
                # each split cluster should have at least one object
                # while True:
                for _ in range(20):
                    _mask = np.random.choice(n_obj, n_add_mask, replace=False)
                    _splits = splits.clone()
                    _splits[_mask] = 0  # TODO: consts

                    if (splits == 1).sum() * (splits == 2).sum() > 0:
                        # at least there is one object in each cluster
                        break
                splits = _splits.clone()
            data['splits'] = splits
        return data


class ClevrImgDataset(Dataset):
    """
    CLEVR Image Only dataset for pytorch
    This is used to pre-save resnet image vector

    TODO: docstring

    Attributes
    ----------
    """

    def __init__(self, image_dir, transform=None):
        """
        Parameters
        ----------
        image_dir : str
            The image directory path of CLEVR
        """
        super().__init__()

        self.image_dir = image_dir
        self.image_filenames = load_image_fnames(self.image_dir)
        self.n_images = len(self.image_filenames)
        self.transform = transform

        logger.info(f'The number of images : {self.n_images}')

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        """
        Parameters
        ----------
        idx : int
            index of question
        """
        image_filename = os.path.join(
            self.image_dir, self.image_filenames[idx]
        )

        image = Image.open(image_filename).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, image_filename


# ------------------------------
# Functions
# ------------------------------
def get_dataloader(
        args,
        multi_q_ver=False,
        raw_image=False,
        resize_shape=[224, 224],
):
    """
    Get dataloader for train and valid

    Parameters
    ----------
    args : ArgumentParser
    multi_q_ver : bool, default is False
        If the argument is True, use Dataset [MultiQAClevrDataset]
    raw_image : bool, default is False
        When True, load raw image itself
    resize_shape : arraylike(height, width), default is [224, 224]
        Image will be resized into this shape

    Returns
    -------
    train_loader : torch Dataloader
    val_loader : torch Dataloader
    """

    # transforms
    # configuration is same with the following repository:
    # https://github.com/mesnico/RelationNetworks-CLEVR
    if raw_image:
        train_transforms = transforms.Compose([
            transforms.Resize(resize_shape),
            transforms.Pad(8),
            transforms.RandomCrop(resize_shape),
            transforms.RandomRotation(2.8),  # 0.5 Rad
            transforms.ToTensor(),
        ])
        val_transforms = transforms.Compose([
            transforms.Resize(resize_shape),
            transforms.ToTensor(),
        ])
    else:
        train_transforms = None
        val_transforms = None

    # dataset
    root = 'data/'
    img_v = args.image_data_version
    data_v = args.question_data_version

    if args.proposed_model_mode:
        sf_t_path = f'results/spatial_features_{img_v}/train_spatial_info.pkl'
        sf_v_path = f'results/spatial_features_{img_v}/val_spatial_info.pkl'
    else:
        sf_t_path = None
        sf_v_path = None

    dataset = MultiQAClevrDataset if multi_q_ver else ClevrDataset

    train_dataset = dataset(
        os.path.join(root, img_v, 'images/train'),
        os.path.join(root, data_v, 'savedata/train_questions.h5'),
        os.path.join(root, img_v, 'scenes/CLEVR_train_scenes.json'),
        args.load_train_context_feature,
        args.train_prefix,
        spatial_feature_path=sf_t_path,
        obj_dir=os.path.join(root, img_v, 'images/cropped_image/train/'),
        obj_feature_path=args.load_train_cropped_feature,
        data_size_ratio=args.data_size_ratio,
        guesser_obj_feature_path=args.load_guesser_train_obj_feature,
        top_k=args.proposed_q_gen_top_k,
        dialogue_h5_path=args.dialogue_h5_train_path,
        raw_image=raw_image,
        transform=train_transforms,
    )

    val_dataset = dataset(
        os.path.join(root, img_v, 'images/val'),
        os.path.join(root, data_v, 'savedata/val_questions.h5'),
        os.path.join(root, img_v, 'scenes/CLEVR_val_scenes.json'),
        args.load_val_context_feature,
        args.val_prefix,
        spatial_feature_path=sf_v_path,
        obj_dir=os.path.join(root, img_v, 'images/cropped_image/val/'),
        obj_feature_path=args.load_val_cropped_feature,
        data_size_ratio=args.data_size_ratio,
        guesser_obj_feature_path=args.load_guesser_val_obj_feature,
        top_k=args.proposed_q_gen_top_k,
        dialogue_h5_path=args.dialogue_h5_val_path,
        raw_image=raw_image,
        transform=val_transforms,
    )

    train_shuffle = False if args.train_not_shuffle else True
    logger.info(f'Train dataloader shuffle state : {train_shuffle}')

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_shuffle,
        num_workers=args.num_workers,
        pin_memory=not(args.not_pin_memory),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=not(args.not_pin_memory),
    )

    return train_loader, val_loader


def get_test_loader(
        args,
        multi_q_ver=False,
        raw_image=False,
        resize_shape=[224, 224],
):
    """
    Get dataloader for evaluation (test)

    Parameters
    ----------
    args : ArgumentParser
    multi_q_ver : bool, default is False
    raw_image : bool, default is False
    resize_shape : arraylike(height, width), default is [224, 224]

    Returns
    -------
    test_loader : torch Dataloader
    """
    if raw_image:
        test_transforms = transforms.Compose([
            transforms.Resize(resize_shape),
            transforms.ToTensor(),
        ])
    else:
        test_transforms = None

    # dataset
    root = 'data/'
    img_v = args.image_data_version
    data_v = args.question_data_version

    if args.proposed_model_mode:
        sf_path = f'results/spatial_features_{img_v}/test_spatial_info.pkl'
    else:
        sf_path = None

    dataset = MultiQAClevrDataset if multi_q_ver else ClevrDataset

    test_dataset = dataset(
        os.path.join(root, img_v, 'images/test'),
        os.path.join(root, data_v, 'savedata/test_questions.h5'),
        os.path.join(root, img_v, 'scenes/CLEVR_test_scenes.json'),
        args.load_test_context_feature,
        args.test_prefix,
        spatial_feature_path=sf_path,
        obj_dir=os.path.join(root, img_v, 'images/cropped_image/test/'),
        obj_feature_path=args.load_test_cropped_feature,
        data_size_ratio=args.data_size_ratio,
        guesser_obj_feature_path=args.load_guesser_test_obj_feature,
        top_k=args.proposed_q_gen_top_k,
        dialogue_h5_path=args.dialogue_h5_test_path,
        raw_image=raw_image,
        transform=test_transforms,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=not(args.not_pin_memory)
    )

    return test_loader


def get_rl_dataloader(args):
    """
    Return torch.utils.Dataloader for RL

    Parameters
    ----------
    args : ArgumentParser

    Returns
    -------
    train_loader : torch.utils.Dataloader
    val_laoder : torch.utils.Dataloader
    """
    root = 'data/'
    img_v = args.image_data_version

    if args.proposed_model_mode:
        train_spatial_feature_path = \
            f'results/spatial_features_{img_v}/train_spatial_info.pkl'
        val_spatial_feature_path = \
            f'results/spatial_features_{img_v}/val_spatial_info.pkl'
    else:
        train_spatial_feature_path = None
        val_spatial_feature_path = None

    train_dataset = RlClevrDataset(
        args.proposed_model_mode,
        os.path.join(root, img_v, 'images/train'),
        os.path.join(root, img_v, args.scene_train_path),
        args.load_train_context_feature,
        args.train_prefix,
        os.path.join(root, img_v, 'images/cropped_image/train/'),
        args.load_train_cropped_feature,
        data_size_ratio=args.data_size_ratio,
        spatial_feature_path=train_spatial_feature_path,
        guesser_obj_feature_path=args.load_guesser_train_obj_feature,
    )

    val_dataset = RlClevrDataset(
        args.proposed_model_mode,
        os.path.join(root, img_v, 'images/val'),
        os.path.join(root, img_v, args.scene_val_path),
        args.load_val_context_feature,
        args.val_prefix,
        os.path.join(root, img_v, 'images/cropped_image/val/'),
        args.load_val_cropped_feature,
        data_size_ratio=args.data_size_ratio,
        spatial_feature_path=val_spatial_feature_path,
        guesser_obj_feature_path=args.load_guesser_val_obj_feature,
    )

    train_shuffle = False if args.train_not_shuffle else True
    logger.info(f'Train dataloader shuffle state : {train_shuffle}')

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_shuffle,
        num_workers=args.num_workers,
        pin_memory=not(args.not_pin_memory),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=not(args.not_pin_memory),
    )

    return train_loader, val_loader


def get_rl_test_loader(args):
    """
    Return torch.utils.Dataloader for RL evaluation

    Parameters
    ----------
    args : ArgumentParser

    Returns
    -------
    test_loader : torch.utils.Dataloader
    """
    root = 'data/'
    img_v = args.image_data_version

    if args.proposed_model_mode:
        test_spatial_feature_path = \
            f'results/spatial_features_{img_v}/test_spatial_info.pkl'
    else:
        test_spatial_feature_path = None

    test_dataset = RlClevrDataset(
        args.proposed_model_mode,
        os.path.join(root, img_v, 'images/test'),
        os.path.join(root, img_v, args.scene_test_path),
        args.load_test_context_feature,
        args.test_prefix,
        os.path.join(root, img_v, 'images/cropped_image/test/'),
        args.load_test_cropped_feature,
        data_size_ratio=args.data_size_ratio,
        spatial_feature_path=test_spatial_feature_path,
        guesser_obj_feature_path=args.load_guesser_test_obj_feature,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=not(args.not_pin_memory),
    )

    return test_loader


if __name__ == '__main__':
    # DEBUG Purpose only
    # When running following code, modify temporary `from . import utils` to
    # `import utils`
    print('>>> Dataset Check')
    dataset = ClevrDataset(
        'data/CLEVR_v1.0/images/val',
        'data/A10/savedata/val_questions.h5',
        'data/CLEVR_v1.0/scenes/CLEVR_val_scenes.json',
        'results/image_vec/val/resnet101_23.npz',
        'CLEVR_val_',
        spatial_feature_path='results/spatial_features/val_spatial_info.pkl',
        obj_dir='data/CLEVR_v1.0/images/cropped_image/val/',
        obj_feature_path='results/image_vec/cropped_images/val/resnet18_7.npz',
        guesser_obj_feature_path=None,
        dialogue_h5_path=None,
        raw_image=None,
        transform=None
    )

    loader = DataLoader(dataset, 1024, shuffle=False)

    for data in loader:
        for k, v in data.items():
            try:
                print(f'{k} : {v.shape}')
            except AttributeError:
                print(k)
        break
