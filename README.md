# Visuo-Skin (ViSk)
Accompanying code for training VisuoSkin policies as described in the paper:
[Learning Precise, Contact-Rich Manipulation through Uncalibrated Tactile Skins](https://visuoskin.github.io)
<br><br>
<p align="center">
<img src="https://github.com/user-attachments/assets/4dcc3819-57c9-4059-889d-e1123e816090" align="center" alt="fig1" />
</p>

## About

ViSk is a framework for learning visuotactile policies for fine-grained, contact-rich manipulation tasks. ViSk uses a transformer-based architecture in conjunction with [AnySkin](https://any-skin.github.io) and presents a significant improvement over vision-only policies as well as visuotactile policies that use high-dimensional tactile sensors like DIGIT.

## Installation
1. Clone this repository
```
git clone https://github.com/raunaqbhirangi/visuoskin.git
```

2. Create a conda environment and install dependencies

```
conda create -f env.yml
pip install -r requirements.txt
```

3. Move raw data to your desired location and set `DATA_DIR` in `utils.py` to point to this location. Similarly, set `root_dir` in `cfgs/local_config.yaml`.

4. Process data for the `current-task` (name of the directory containing demonstration data for the current task) and convert to pkl.

```
python process_data.py -t current-task
python convert_to_pkl.py -t current-task
```
5. Install `xarm-env` using `pip install -e envs/xarm-env`

6. Run BC training
```
python train_bc.py 'suite.task.tasks=[current-task]'
```
