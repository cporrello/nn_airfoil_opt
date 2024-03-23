# nn_airfoil_opt
Code repo for project regarding design optimization with neural networks:
- `Geo-FNO/`: version of [Geo-FNO code](https://github.com/neuraloperator/Geo-FNO) with minimal modifications (changes described in paper) 
- `dnn/`: directory containing Python code for training the fully-connected feedforward network (FNN) described in the paper
- `models/`: directory containing trained FNN and Geo-FNO models and scripts for post-training analysis
- `optim/`: directory containing Python code for solving the optimization problem described in the paper
- `wandb/`: duplicate copy of `dnn/` but with modifications to perform Weights and Biases sweeps of the FNN design
