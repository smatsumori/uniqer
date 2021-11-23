import os
import gdown
import subprocess

# logging
from logging import getLogger
logger = getLogger(__name__)

datasets = [
    (
        # CLEVR Ask3 dataset
        ['1KLMcIR60P-L7jlHQg2IFpDgENHJOW6AI'],
        './data/CLEVR_Ask3.tar.gz',
        './data'
    ),
    (
        # CLEVR Ask4 dataset
        ['14_HKTbcJT1l9tUY_cDDMeJ6bC10QDUnn'],
        './data/CLEVR_Ask4.tar.gz',
        './data'
    ),
    (
        # guesswhat train
        ['1-PLlTN4ux5mDsL56hjGIBzp7v8klwyPi'],
        './data/guesswhat.tar.gz',
        './data'
    ),
]
for dids, path, dest in datasets:
    for did in dids:
        if os.path.exists(path):
            break
        url = 'https://drive.google.com/uc?id={}'.format(did)
        gdown.download(url, path, quiet=False)
        if os.path.exists(path):
            print(f'{path} already exists! skipping...')
            break
        else:
            logger.info('Trying other mirrors.')
    command = ['tar', '-xf', path, '-C', dest]
    logger.info('Running subprocess: {}'.format(' '.join(command)))
    r = subprocess.run(command, check=True)
    logger.info('Command completed: {}'.format(r))
