import os
import asyncio
import imgkit
import pdfkit
from joblib import Parallel, delayed
from pyppeteer import launch
from jinja2 import Environment, FileSystemLoader

# logging
from logging import getLogger
logger = getLogger(__name__)

# set environment
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# Jinja settings
file_loader = FileSystemLoader('./')

# pyppeteer settings
BROWSER_SETTINGS = {
    'logLevel': 'ERROR',
    'args': [
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--disable-dev-shm-usage'
    ]
}
PPT_PDF_SETTINGS = {
    'format': 'A3',
    'printBackground': True,
    'margin': {
      'top': 0,
      'bottom': 0,
      'left': 0,
      'right': 0
    }
}
PPT_JPG_SETTINGS = {
    'fullPage': True
}

# Render options for imgkit
WK_JPG_SETTINGS = {
    'enable-local-file-access': None,
    'width': 860,
    'disable-smart-width': '',
    'quiet': '',
}

# Render options for pdfkit
WK_PDF_OPTIONS = {
    'enable-local-file-access': None,
    # 'orientation': 'Landscape',
    'zoom': 0.70,
    'quiet': '',
}


def summarize_dialogue(
    dia_summary, mode, epoch, image_base_path, save_path,
    limit_save_idxs=[], summary_saver='wkhtmltopdf'
):
    """Summarize and save dialogue information to .pdf and .jpg files.

    Parameters
    ----------
    dia_summary : list
        Dialogue summary.
    mode : str
        mode in ['train', 'valid', 'test_i', 'test_o']
    epoch : int
        epoch
    image_base_path :
        Path to image dataset.
    save_path :
        save_path
    limit_save_idxs :
        This will limit the number of dialogues to save.
    """
    if not limit_save_idxs:
        return [], []

    assert summary_saver in ['wkhtmltopdf', 'pyppeteer']
    logger.info('Generating dialogue summaries (.jpeg and .pdf).')

    def helper_q(q):
        """A helper function for question formatting.

        Parameters
        ----------
        q : str
            A question
        """
        q.replace('<', '')
        q.replace('>', '')
        return q

    def helper_p(p):
        """A helper function for prediction

        Parameters
        ----------
        p : list
            A list of predicted probabilities.
        """
        return ['{:.3f}'.format(round(f, 3)) for f in p]

    summary_list = []
    for sm in dia_summary:
        if 'split_actions' not in sm:
            split_actions = [None for _ in range(len(sm['answers']))]
        else:
            split_actions = sm['split_actions']
        dict_for_template = {
            'file_path': os.path.abspath(
                os.path.join(image_base_path, sm['image_filename'])
            ),
            # TODO: <idk>
            'qas': [
                {
                    'question': helper_q(q),
                    'answer': ['No', 'Yes', 'N/A'][a],
                    'prediction': helper_p(p),
                    'split_action': s
                } for q, a, p, s
                in zip(
                    sm['raw_str_programs'], sm['answers'],
                    sm['predictions'], split_actions
                )
            ],
            'results': sm['results'],
            'target_id': sm['target_id'],
            'submitted_id': sm['submitted_id']
        }
        summary_list.append(dict_for_template)

    # 1. Summary all (in single pdf)
    # Create jinja
    env = Environment(loader=file_loader)
    env.trim_blocks = True
    env.lstrip_blocks = True
    env.rstrip_blocks = True
    # source template paths
    src_path = os.path.join('./src/utils/rl', 'templates')
    tl_summary_all = env.get_template(
        os.path.join(src_path, 'summary_all.html')  # for all
    )
    # create summary save path
    summary_save_path = os.path.join(save_path, f'summary_{mode}')
    os.makedirs(summary_save_path, exist_ok=True)

    # render summary all
    rendered_all = tl_summary_all.render(
        # TODO: unlock limit
        mode=mode, epoch=epoch,
        dials=[summary_list[idx] for idx in limit_save_idxs]
    )
    with open(os.path.join(summary_save_path, 'summary_all.html'), 'w') as f:
        f.write(rendered_all)

    all_artifact_path = os.path.join(
        summary_save_path, 'summary-all-{}-epoch-{:03d}.pdf'.format(
            mode, epoch
        )
    )
    # Generate pdf
    if summary_saver == 'wkhtmltopdf':
        pdfkit.from_file(
            os.path.join(summary_save_path, 'summary_all.html'),  # src
            all_artifact_path,
            options=WK_PDF_OPTIONS
        )
    elif summary_saver == 'pyppeteer':
        loop = asyncio.get_event_loop()
        loop.run_until_complete(
            generate_pdf(
                os.path.join(summary_save_path, 'summary_all.html'),
                all_artifact_path,  # destination path
            )
        )

    # 2. Individual summary
    indv_artifact_paths = []

    def wk_helper(idx):
        """ Helper function for executing imgkit in parallel.
        """
        # Create jinja environment.
        _env = Environment(loader=file_loader)
        _env.trim_blocks = True
        _env.lstrip_blocks = True
        _env.rstrip_blocks = True
        tl_summary_indv = _env.get_template(
            os.path.join(src_path, 'summary.html')  # for individual
        )
        file_name = 'summary-{:03d}-{}-epoch-{:03d}'.format(
            idx, mode, epoch
        )
        summary_html = os.path.join(
            summary_indv_save_path, file_name+'.html'
        )
        rendered_indv = tl_summary_indv.render(
            mode=mode, epoch=epoch, dial=summary_list[idx]
        )
        with open(summary_html, 'w') as f:
            f.write(rendered_indv)
        dest_path = os.path.join(
            summary_indv_save_path, file_name+'.jpg'
        )
        imgkit.from_file(
            summary_html, dest_path, options=WK_JPG_SETTINGS
        )
        return dest_path

    if limit_save_idxs:
        logger.info(f'Generating {len(limit_save_idxs)} indiv images.')
        # create summary save path (individual)
        summary_indv_save_path = os.path.join(summary_save_path, 'batches')
        os.makedirs(summary_indv_save_path, exist_ok=True)
        try:
            if summary_saver == 'wkhtmltopdf':
                indv_artifact_paths = Parallel(
                    # NOTE: Too many jobs may degrade performance.
                    n_jobs=len(limit_save_idxs), backend='threading'
                )(
                    delayed(wk_helper)(i) for i in limit_save_idxs
                )
            elif summary_saver == 'pyppeteer':
                src, dst = [], []
                tl_summary_indv = env.get_template(
                    os.path.join(src_path, 'summary.html')  # for individual
                )
                # Generate individual html
                for idx in limit_save_idxs:
                    file_name = 'summary-{:03d}-{}-epoch-{:03d}'.format(
                        idx, mode, epoch
                    )
                    summary_html = os.path.join(
                        summary_indv_save_path, file_name+'.html'
                    )
                    rendered_indv = tl_summary_indv.render(
                        mode=mode, epoch=epoch, dial=summary_list[idx]
                    )
                    with open(summary_html, 'w') as f:
                        f.write(rendered_indv)
                    dest_path = os.path.join(
                        summary_indv_save_path, file_name+'.jpg'
                    )
                    src.append(summary_html)
                    dst.append(dest_path)
                    indv_artifact_paths.append(dest_path)
                # Generate .jpeg
                loop = asyncio.get_event_loop()
                loop.run_until_complete(generate_jpegs(src, dst))
        except IndexError:
            logger.error(f'Index {idx} not registered!')

    return all_artifact_path, indv_artifact_paths


async def generate_pdf(source_path, dest_path):
    # Convert to absoute path
    source_path = os.path.abspath(source_path)
    dest_path = os.path.abspath(dest_path)
    source_path = 'file://' + source_path
    settings = PPT_PDF_SETTINGS.copy()
    settings['path'] = dest_path
    browser = await launch(BROWSER_SETTINGS)
    page = await browser.newPage()
    await page.goto(source_path, {'waitUntil': 'networkidle2'})
    await page.pdf(settings)
    await browser.close()


async def generate_jpegs(src_paths, dst_path):
    browser = await launch(BROWSER_SETTINGS)

    async def gen_jpg(browser, src, dst):
        # Create settings
        src = os.path.abspath(src)
        dst = os.path.abspath(dst)
        src = 'file://' + src
        settings = PPT_JPG_SETTINGS.copy()
        settings['path'] = dst
        page = await browser.newPage()
        await page.goto(src)
        await page.screenshot(settings)
    cors = [
        gen_jpg(browser, s, d) for s, d in zip(src_paths, dst_path)
    ]
    await asyncio.gather(*cors)
    await browser.close()


def test_summarize_dialogue():
    image_base_path = './data/CLEVR_v1.0/images/val'
    save_path = './test_tmp'  # will be removed afeter the test
    os.makedirs(save_path)  # exist_ok = False
    sm = {
        'image_filename': 'CLEVR_val_000000.png',
        'raw_str_programs':
        [
            'This is a pen.',
            'I have a pen.',
            'This is Ben.',
            'I hate Ben.',
        ],
        'answers':
        [
            0, 1, 1, 2
        ],
        'predictions':
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        'results': True,
        'target_id': 0,
        'submitted_id': 0,
    }

    # summary for splitnetwork
    sm_sp = {
        'image_filename': 'CLEVR_val_000001.png',
        'raw_str_programs':
        [
            'This is a pen.',
            'I have a pen.',
            'This is Ben.',
            'I hate Ben.',
        ],
        'answers':
        [
            0, 1, 1, 2
        ],
        'predictions':
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        'split_actions':
        [
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
        ],
        'results': True,
        'target_id': 0,
        'submitted_id': 0,
    }
    dia_summary = [
        sm, sm, sm, sm_sp, sm_sp
    ] + [sm for _ in range(10)] + [sm_sp for _ in range(20)]
    summarize_dialogue(
        dia_summary, 'valid', 100, image_base_path, save_path,
        limit_save_idxs=[i for i in range(len(dia_summary))],
        summary_saver='pyppeteer'
    )
