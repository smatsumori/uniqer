import sys
import datetime

from logging import getLogger, StreamHandler, FileHandler, Formatter
from logging import INFO, DEBUG

from .utils import is_jupyter_notebook

# Parameters
now = datetime.datetime.now()
# save_filename = 'results/text_logs/' + now.strftime('%Y%m%d_%H') + '.log'
save_filename = 'results/text_logs/' + now.strftime('%Y%m%d') + '.log'

if is_jupyter_notebook():
    print('Jupyter Notebook Mode detected. Do not saving log text file.')
    disable_stream_handler = False
    disable_file_handler = True
else:
    disable_stream_handler = False
    disable_file_handler = False

# default logging format
datefmt = '%Y/%m/%d %H:%M:%S'
default_fmt = Formatter(
    '[%(asctime)s.%(msecs)03d] %(levelname)5s '
    '(%(process)d) %(filename)s: %(message)s',
    datefmt=datefmt
)

# level: CRITICAL > ERROR > WARNING > INFO > DEBUG
logger = getLogger()
logger.setLevel(INFO)

# set up stream handler
if not disable_stream_handler:
    try:
        # Rainbow Logging
        from rainbow_logging_handler import RainbowLoggingHandler
        color_msecs = ('green', None, True)
        stream_handler = RainbowLoggingHandler(
            sys.stdout, color_msecs=color_msecs, datefmt=datefmt
        )
        # msecs color
        stream_handler._column_color['.'] = color_msecs
        stream_handler._column_color['%(asctime)s'] = color_msecs
        stream_handler._column_color['%(msecs)03d'] = color_msecs
    except Exception:
        stream_handler = StreamHandler()

    stream_handler.setFormatter(default_fmt)
    stream_handler.setLevel(DEBUG)
    logger.addHandler(stream_handler)

if not disable_file_handler:
    file_handler = FileHandler(filename=save_filename)
    file_handler.setFormatter(default_fmt)
    file_handler.setLevel(DEBUG)
    logger.addHandler(file_handler)


def set_logger_level(logger, level):
    """
    Parameters
    ----------
    logger
    level: str  (['INFO', 'DEBUG'])
    """
    if level == 'INFO':
        logger.setLevel(INFO)
    elif level == 'DEBUG':
        logger.setLevel(DEBUG)
    return logger
