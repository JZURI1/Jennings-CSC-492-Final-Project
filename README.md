# URI-CSC-492-Final-Project-2026

## Analysis of: AttackBench: Evaluating Gradient-based Attacks for Adversarial Examples
Javon Jennings \
4/29/2026 \
URI CSC 492 \
Final Project 

## Student anaysis of \
@inproceedings{cina2025attackbench,
  title={Attackbench: Evaluating gradient-based attacks for adversarial examples},
  author={Cin{\`a}, Antonio Emanuele and Rony, J{\'e}r{\^o}me and Pintor, Maura and Demetrio, Luca and Demontis, Ambra and Biggio, Battista and Ayed, Ismail Ben and Roli, Fabio},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={3},
  pages={2600--2608},
  year={2025},
  DOI={10.1609/aaai.v39i3.32263}
}

## About
In this project I implemented the AttackBench from the paper on my own device using VS code running with Unity. My goal was to recreate the results of the paper for the ℓ 2 model running within the constraints of the resources I had at my disposal. Below is a step by step guide as well as requirements for running AttackBench if you are working with similar resources at your disposal.

## Step by Step Guide
1. Activate Unity 
```
ssh unity 
```
2. request GPU 
```
srun --partition=gpu --gres=gpu:a100:1 --mem=32G --time=06:00:00 --pty bash
```
3. Load models 
```
module load gcc/11.2.0 
module load cuda/11.8 
module load conda/latest 
```

4. Load AttackBench from Github 
```
git clone https://github.com/attackbench/attackbench.git
```
5. Open File
```
cd attackbench
```
6. Activate conda environment 
```
conda env create -f environment.yml 
```
7. Activate Attackbench
```
conda activate atkbench
```
8. Run the Attackbench
```
python -m attack_evaluation.run  -F results_dir/ with model.augustin_2020 attack.adv_lib_fmn attack.threat_model="l2" dataset.num_samples=1000 dataset.batch_size=64 seed=42
```

## Software Installation
VS Code with Python installed \
Access to Unity for GPU \
GCC version 11.2.0 \
CUDA version 11.8 \
Conda version miniforge 3-24.7.1 


## Models

## System Requirements
GPU: A100 recommended
RAM: 32
Other: Bash to run Linux for Unity


## Contact
For questions about the project or this repository email javon_jennings@uri.edu
