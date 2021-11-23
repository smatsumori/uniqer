import os
import random

from . import question_engine as qeng
from utils.utils import load_vocab, load_json
import modules.question_generator.program_manipulates_func as pmf
from utils.consts import TOKEN_LABEL_CLEVR
from utils.consts import TOKEN_LABEL_ASK3
from utils.consts import TOKEN_LABEL_ASK4
from utils.consts import TOKEN_LABEL_SPECIAL, TOKEN_LABEL_RELATE

# logging
from logging import getLogger
logger = getLogger(__name__)


class Oracle():
    """
    Oracle
        Randomly selecting one reference object for each image in the dataset.
        There are no training parameters, so basically, unless you want to
        reset the reference object, you only need to create it once.

    Attributes
    ----------
    metadata : dict
        CLEVR dataset metadata
    scenes : dict
        CLEVR dataset scene data
    ref_objects : dict
        {image_filename: object_id}
    param_name_to_type : dict
        {parameter : type}  --> ex. {'cube' : Shape}  (maybe.)
    image_name_to_scene : dict
        {image_filename: scene}
    """

    def __init__(
            self,
            metadata,
            scenes,
            vocab_path,
            image_data_version,
            restrict_objects=None,
            idk_mode=False,
    ):
        """
        Parameters
        ----------
        metadata : str or dict
            CLEVR dataset metadata or metadata json file path
        scenes : str or dict
        vocab_path : str
        image_data_version : str
            The version of image data.
        restrict_objects : list of list. Sublists contain allowed
            objects when selecting targets in `_init_ref_objects`.
            shape: (n_scenes, n_objects_to_restrict)
        idk_mode : bool, default is False
        """
        self.metadata = load_json(metadata)
        self.scenes = load_json(scenes)
        self.vocab = load_vocab(vocab_path)

        self.restrict_objects = restrict_objects
        self.ref_objects = self._init_ref_objects()

        self.param_name_to_type = self._init_param_name_to_type()
        self.image_name_to_scene = self._init_image_name_to_scene()

        if self.restrict_objects is not None:
            n_rest = len(sum(self.restrict_objects, []))
            logger.warning('Number of candidate targets: {}'.format(n_rest))
        else:
            logger.info('Restrcited Objects data is not specified '
                        '(maybe validation mode ?)')

        logger.info(f'Number of images: {len(self.image_name_to_scene)}')

        self.idk_mode = idk_mode
        if self.idk_mode:
            logger.info('IDK mode enabled (Oracle)')

            # update token labels
            def _update_token_label(original):
                new = []
                for token in original:
                    if token in self.vocab['program_token_to_idx']:
                        new.append(token)
                return new
            if image_data_version == 'CLEVR_Ask3':
                tk = TOKEN_LABEL_ASK3
            elif image_data_version == 'CLEVR_Ask4':
                tk = TOKEN_LABEL_ASK4
            else:
                tk = TOKEN_LABEL_CLEVR
            self.tl_color = _update_token_label(tk.TOKEN_LABEL_COLOR)
            self.n_color = len(self.tl_color)
            self.tl_size = _update_token_label(tk.TOKEN_LABEL_SIZE)
            self.n_size = len(self.tl_size)
            self.tl_material = _update_token_label(tk.TOKEN_LABEL_MATERIAL)
            self.n_material = len(self.tl_material)
            self.tl_shape = _update_token_label(tk.TOKEN_LABEL_SHAPE)
            self.n_shape = len(self.tl_shape)

    def _init_ref_objects(self, verbose=True):
        """
        Select an object for each scene (image) and store it by object_id.
        Target objects will be slected from the objects stored in
        `self.restrict_objects`.

        Parameters
        ----------
        verbose : bool, default is True
        """
        scenes = self.scenes['scenes']
        if self.restrict_objects is not None:
            # Traget objects will be selected from
            # the restricted object list
            if verbose:
                logger.info('Ignore objects enabled.')
            assert len(self.restrict_objects) == len(scenes)
            ocands = self.restrict_objects
        else:
            # No restriction for candidates if not restricted_objects
            ocands = [list(range(len(s['objects']))) for s in scenes]

        ref_objects = {}
        for oc, sc in zip(ocands, scenes):
            # select the target object from the object candidates
            if len(oc) > 0:
                ref_objects[sc['image_filename']] = random.choice(oc)
            else:
                # set -1 for no candidates scene ?  --> skip in obj_test
                ref_objects[sc['image_filename']] = -1
        return ref_objects

    def reset_ref_objects(self):
        """
        Reset reference objects (supposed to be called each epoch when RL)
        TODO: Maybe we should consider previous `ref_objects`
        """
        self.ref_objects = self._init_ref_objects(verbose=False)
        logger.info('Oracle resetted the reference objects')

    def _init_param_name_to_type(self):
        param_name_to_type = {}
        for k, v in self.metadata['types'].items():
            if v is None:
                continue
            elif type(v) is list or type(v) is tuple:
                for _v in v:
                    param_name_to_type[_v] = k
            else:
                param_name_to_type[v] = k
        return param_name_to_type

    def _init_image_name_to_scene(self):
        image_name_to_scene = {}
        for scene in self.scenes['scenes']:
            image_name_to_scene[scene['image_filename']] = scene
        return image_name_to_scene

    def _return_idk(self, program, alpha=1.0):
        """
        Return IDK or not (IDK probability is hard-coded currently)
            - shape   : 1/3
            - material: 1/2
            - size    : 1/2
            - color   : 1/8
            If the composed question is asked such as "Is it blue cube?",
            IDK probability will be (1/3) * (1/8) * alpha

        Parameters
        ----------
        program : list
            list of dicts (CLEVR functionnal program)
        alpha : float, default is 1.0
            coefficient (for future usage)

        Returns
        -------
        idk : bool
        """
        idk_prob = 1.0
        done = []

        for token_id in program:
            token = self.vocab['program_idx_to_token'][token_id]
            if token in done:
                pass
            elif token in self.tl_color:
                idk_prob *= 1 / self.n_color
            elif token in self.tl_size:
                idk_prob *= 1 / self.n_size
            elif token in self.tl_material:
                idk_prob *= 1 / self.n_material
            elif token in self.tl_shape:
                idk_prob *= 1 / self.n_shape
            # TODO
            elif token in TOKEN_LABEL_SPECIAL:
                pass
            elif token in TOKEN_LABEL_RELATE:
                pass
            else:
                logger.error(f'Unknown token <{token}>. Fix this.')

            done.append(token)

        if random.random() < idk_prob:
            idk = True
        else:
            idk = False

        return idk * alpha

    def _answering_question(self, image_filename, program, require_pp=False):
        """
        Parameters
        ----------
        image_filename : str
        program : list
            list of dicts (CLEVR functionnal program)
        require_pp : bool, default is False
            requiring pre-process or not

        Returns
        -------
        out : list
            answer id
        """
        if require_pp:
            program = self.question_tokens_to_program(program)

        # Gross.
        for f in program:
            try:
                if len(f['value_inputs']) != 0:
                    f['side_inputs'] = f['value_inputs']
                del f['value_inputs']
            except KeyError:
                # when value_inputs does not exist
                # (have already been translated into `side_inputs`)
                pass

        scene = self.image_name_to_scene[image_filename]

        try:
            for i in range(len(program)):
                if i == 0:
                    f = [program[0]]
                else:
                    f.append(program[i])
                f[i]['_output'] = qeng.answer_question(
                    {'nodes': f},
                    scene,
                    all_outputs=True
                )[-1]
            # The answer to the last function is the answer to the entire
            # question
            return f[-1]['_output']
        except qeng.InvalidQuestionError:
            # TODO: logging original_program
            return [-1]
        except UnboundLocalError:
            # `f` is not generetad
            return [-1]

    def answering_yn(self, image_filename, program, require_pp=False):
        # should be placed in try block ? In order to detect syntax error
        # of generated question and get NAN in that case.
        # Or maybe separete 'NAN' and 'SYNTAX' ?
        """
        Parameters
        ----------
        image_filename : str
        program : list
            list of dicts (CLEVR functionnal program) or
            list of token index (`require_pp` must be True)
        require_pp : bool
            requiring pre-process or not

        Returns
        -------
        <return> : int
            0 represents NO, 1 do YES, 2 do NAN and 3 do IDontKnow
            INFO To embedding answer_id, we do not use -1 for NAN
            * TODO: remove magic no
        """

        if self.idk_mode:
            if self._return_idk(program):
                # For simple Attribute questions,
                # idk is more likely to be returned
                return 3

        answer = self._answering_question(
            image_filename, program, require_pp=require_pp
        )

        if isinstance(answer, bool):
            return 2
        elif -1 in answer:
            # Invalid question
            return 2
        else:
            if self.ref_objects[image_filename] in answer:
                return 1
            else:
                return 0

    def question_tokens_to_program(self, question):
        """
        Preprocessing question

        Parameters
        ----------
        question : list of int
        """
        str_program = pmf.decode(question, self.vocab['program_idx_to_token'])
        return pmf.str_to_program(str_program)

    def meaningful_question(
            self, image_filename, program, n_obj, require_pp=False
    ):
        """
        Whether at least one object has a different answer (y / n)

        Parameters
        ----------
        image_filename : str
        program : list
            list of dicts (CLEVR functionnal program) or
            list of token index (`require_pp` must be True)
        n_obj : int
        require_pp : bool
            requiring pre-process or not

        Returns
        -------
        <return> : bool
            True when meaningful question, else False
        """
        answer = self._answering_question(
            image_filename, program, require_pp=require_pp
        )
        if isinstance(answer, bool):
            return False
        elif -1 in answer:
            # answer is -1 (which means, InvalidQuestionError)
            return False
        elif isinstance(answer, list):
            return False if len(answer) == n_obj or len(answer) == 0 else True
        else:
            # answer is not a list of objects (ex. 'green')
            return False


def debug_update_oracle_refs(args):
    scene_path = os.path.join(
        './data', args.image_data_version, args.scene_train_path
    )
    oracle = Oracle(
        args.metadata_path, scene_path, args.vocab_path,
        args.image_data_version
    )
    for i in range(3):
        for idx_scene, scene in enumerate(oracle.scenes['scenes']):
            print(f"{scene['image_filename']} : "
                  f"{oracle.ref_objects[scene['image_filename']]}")
            if idx_scene == 10:
                break

        oracle.reset_ref_objects()
