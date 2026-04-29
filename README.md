# URI-CSC-492-Final-Project-2026

## Analysis of: AttackBench: Evaluating Gradient-based Attacks for Adversarial Examples
Javon Jennings \
4/29/2026 \
URI CSC 492 \
Final Project 

Student anaysis of \
inproceedings{cina2025attackbench,
  title={Attackbench: Evaluating gradient-based attacks for adversarial examples},
  author={Cin{\`a}, Antonio Emanuele and Rony, J{\'e}r{\^o}me and Pintor, Maura and Demetrio, Luca and Demontis, Ambra and Biggio, Battista and Ayed, Ismail Ben and Roli, Fabio},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={3},
  pages={2600--2608},
  year={2025},
  DOI={10.1609/aaai.v39i3.32263}
}

## Step by Step Guide
1. Activate Unity \
ssh unity \
request GPU \
srun --partition=gpu --gres=gpu:a100:1 --mem=30G --time=06:00:00 --pty bash

2. Load models \
"module load gcc/11.2.0" \
module load cuda/11.8 \
module load conda/latest \

3. Load AttackBench from Github \
git clone https://github.com/attackbench/attackbench.git

5. Activate environment \
conda env create -f environment.yml 


## Software Installation
VS Code with Python Installed 
Access to Unity for GPU 
Install Miniconda 


## Models

## System Requirements
32 GB Ram Unity


## Contact
For questions about the project or this repository email javon_jennings@uri.edu
