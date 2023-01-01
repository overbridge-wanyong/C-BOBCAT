# C-BOBCAT: Constrained Version of Bilevel Optimization-Based Computerized Adaptive Testing
### Environment Setup
This repository uses the following Pytorch version in Python3.
``` bash
torch==1.12.1
```
### Data
You can download the preprocessed datasets from [Google Drive](https://drive.google.com/file/d/18jMoNc12cfngyD796YITRiEp1KIq4oVu/view?usp=sharing) to `/data/` folder. Preprocessing scirpts can be found in `utils/` folder.
### Training
Train C-BOBCAT
``` bash
python train.py
    --dataset mapt-math
    --model binn-biased
    --n_query 8
    --lamda 3e-2
    --cuda
    --gumbel
```
Hyperparameter ranges are:
``` bash
hyperparameters = [
    [('dataset',), ['mapt-math', 'mapt-read']],
    [('model',), ['binn-biased', 'biirt-biased']],
    [('fold',), [ 1, 2, 3, 4, 5 ]],
    [('hidden_dim'), [256]],
    [('lr',), [ 1e-3 ]],
    [('inner_lr',), [ 2e-1, 1e-1, 5e-2]],
    [('meta_lr',), [ 1e-4 ]],
    [('inner_loop',), [ 5 ]],
    [('policy_lr',), [2e-3,  2e-4]],
    [('n_query',), [2, 4, 8]],
    [('lamda',), [ 3e-3, 1e-3, 3e-2, 1e-2]]
]
```

Train IRT-based Model
``` bash
python irt.py
    --dataset {mapt-math, mapt-read}
    --model {irt-active, irt-random}
    --n_query {2, 4, 8}
```

