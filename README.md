# Unified Questioner Transformer for Descriptive Question Generation in Goal-Oriented Visual Dialogue
Pytorch training code for UniQer and CLEVR ask environment.

## Materials
- [arxiv](https://arxiv.org/abs/2106.15550)
- [Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Matsumori_Unified_Questioner_Transformer_for_Descriptive_Question_Generation_in_Goal-Oriented_Visual_ICCV_2021_paper.html)


## Setup
### Requirements
- Ubuntu 16.04 / 18.04 (recommended)
- Python 3.8 (recommended)
- Cuda compatible GPU(s) (recommended)

### Package installation
All required packages can be installed via pipenv.
- Make sure that your `PyTorch` version is compatible with your `cuda` version.
- If you use a different version of `PyTorch` than the one we used(1.2.0), the resnet download path may change.
  (check `ivqg/src/modules/image_encoder/imagenet_model.py`)
```bash
$ pipenv install --skip-lock
$ pipenv shell
(ivqg) $ wandb login  # follow the instruction
---
$ sudo apt install wkhtmltopdf  # for pdf visualization
```


### Download CLEVR Ask datasets
Downloadable version of datasets are available.
- CLEVR Ask3
  - [CLEVR_Ask3(ImageDataset)](https://drive.google.com/file/d/1u0cg0VOWEG3rgLiUrCfU5hu9zAh2ZZiQ/view?usp=sharing) to `data/`
  - [M85k3(Questions)](https://drive.google.com/file/d/11GWQ5jYUdwNahgEb-bvEko29_VkJxH7r/view?usp=sharing) to `data/`
- CLEVR Ask4
  - [CLEVR_Ask4(ImageDataset)](https://drive.google.com/file/d/1yYsF4iHIEsteerIOmAFx8J6tLB4Pp2qA/view?usp=sharing) to `data/`
  - [M85k4(Questions)](https://drive.google.com/file/d/1ooXAoDlDncwloCGkk5aD4d0mesFn9nmA/view?usp=sharing) to `data/`

#### pre-saved vectors
  - [Image vectors of CLEVR_Ask3](https://drive.google.com/file/d/15KIPKR5VJV-6yaqaOi3UI8Mnc9v0YNTW/view?usp=sharing) to `results/`
  - [Image vectors of CLEVR_Ask4](https://drive.google.com/file/d/14duU9Zj15S8lIBdMggQo0BHMVQDYeUqh/view?usp=sharing) to `results/`
  - [Spatial vectors of CLEVR_Ask3](https://drive.google.com/file/d/1cv8bquchXaJsp_chdnAJpHgKuF2tE7Dp/view?usp=sharing) to `results/`
  - [Spatial vectors of CLEVR_Ask4](https://drive.google.com/file/d/1FGAmmC4yqpk1W_XcyoquT5r8jchWxDTK/view?usp=sharing) to `results/`

#### restricted object list
  - [restricted object list of CLEVR_Ask3](https://drive.google.com/file/d/1yzGXtySIGYYI0oeoSB91rCbXgPKCxwep/view?usp=sharing) to `data/M85k3/save_data/`
  - [restricted object list of CLEVR_Ask4](https://drive.google.com/file/d/1Zisw262DOREm1eiIriXKnZiTgq7DJfuQ/view?usp=sharing) to `data/M85k4/save_data/`
  

### Dataset setup(optional)
If you want to setup the dataset manually please see the following document.
- [Dataset setup](docs/SETUP.md)

### Download pre-trained models (optional)
(Pre-)trained models and pre-extracted features are available as:
- [(pre-)trained models](https://drive.google.com/file/d/1Rbd1DOhLGXIXsv38t-SYST4-nciZeFcu/view?usp=sharing)

The extracted folder should be placed under the root of the workspace.


## Training & Evaluations
### Supervised learning
Execute a training in supervised learning(chose either `ask3_uniqer_rl.yaml` or `ask4_uniqer_rl.yaml`)
```bash
$ pipenv run python src/main.py --train_single_tf --yaml_path params/ask3_uniqer_supervised.yaml
```
To evaluate the model, run the following:
```bash
$ pipenv run python src/main.py --check_single_tf --yaml_path params/ask3_uniqer_supervised.yaml
```

### Reinforcement learning
Execute a training in reinforcement learning (chose either `ask3_uniqer_rl.yaml` or `ask4_uniqer_rl.yaml`)
```bash
$ pipenv run python src/main.py --train_rl --yaml_path params/ask3_uniqer_rl.yaml
```

To evaluate the model, run the following:
```bash
$ pipenv run python src/main.py --check_rl --yaml_path params/ask3_uniqer_rl.yaml
```

## Experimental Results
### The exact number of training and evaluation runs
- Supervised Learning: 587 epochs (set 50epochs patience / 1000 epochs)
- Reinforcement Learning: 150 epochs

### A description of results with central tendency (e.g. mean) & variation (e.g. error bars)
- Supervised Learning
  - For both Ask3 and Ask4 dataset, Uniqer was able to detect objects that match a given dialogue with near-perfect F1 score.
  - As for the QDT performance, the correct address ratio is higher than the perfect address ratio for both datasets.

    | Model        | F1 score | Perfect Address     | Correct Address     |
    | :----------: | :------: | :-----------------: | :-----------------: |
    | UniQer(Ask3) | 0.994    | 57.67 %             | 86.91 %             |
    | UniQer(Ask4) | 0.994    | 43.20 %             | 69.79 %             |

- Reinforcement Learning
  - Ask3 Dataset

	| Model Name                    | New Image Task Success (%) | New Object Task Success (%) |
	| :---------------------------: | :------------------------: | :-------------------------: |
	| Baseline                      | 60.00 ± 6.35               | 59.60 ± 6.87                |
	| Ours(vanilla)                 | 72.98 ± 3.13               | 72.88 ± 3.47                |
	| Ours(not unified MLP Guesser) | 69.43 ± 2.75               | 69.50 ± 2.99                |
	| Ours(not unified)             | 50.61 ± 6.51               | 50.37 ± 6.02                |
	| Ours(full)                    | 84.10 ± 4.41               | 83.96 ± 4.70                |

  - Ask4 Dataset

	| Model Name                    | New Image Task Success (%) | New Object Task Success (%) |
	| :---------------------------: | :------------------------: | :-------------------------: |
	| Baseline                      | 64.75 ± 0.82               | 64.21 ± 0.34                |
	| Ours(vanilla)                 | 67.38 ± 4.18               | 67.01 ± 4.34                |
	| Ours(not unified MLP Guesser) | 72.89 ± 5.95               | 72.35 ± 5.94                |
	| Ours(not unified)             | 65.15 ± 3.33               | 64.25 ± 3.01                |
	| Ours(full)                    | 81.20 ± 4.37               | 80.50 ± 4.86                |


### The average runtime for each result, or estimated energy cost and a description of the computing infrastructure used
  - Supervised Learning: 25h (with a Quadro RTX 8000)
  - Reinforcement Learning: 20h (with a Quadro RTX 8000)


## Repository Contributers
Issues and pull requests are always welcomed!
- [@Kosuke Shingyouchi](https://github.com/sngyo)
- [@Shoya Matsumori](https://github.com/smatsumori)
- [@Soma Kanazawa](https://github.com/soma-knzw)
