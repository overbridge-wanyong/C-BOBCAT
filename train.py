from math import exp
import numpy as np
from scipy.sparse.sputils import matrix
import torch
import os
from dataset import Dataset, collate_fn
from utils.utils import compute_auc, compute_accuracy, data_split, batch_accuracy
from model import MAMLModel
from policy import PPO, Memory, StraightThrough
from copy import deepcopy
from utils.configuration import create_parser, initialize_seeds
import time
import os
import statistics
import subprocess
from collections import defaultdict
DEBUG = False if torch.cuda.is_available() else True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_val_score, best_test_score = 0, 0
best_val_auc, best_test_auc = 0, 0
best_epoch = -1
run = None

def clone_meta_params(batch):
    return [meta_params[0].expand(len(batch['input_labels']),  -1).clone(
    )]


def inner_algo(batch, config, new_params, create_graph=False):
    for _ in range(params.inner_loop):
        config['meta_param'] = new_params[0]
        res = model(batch, config)
        loss = res['train_loss']
        grads = torch.autograd.grad(
            loss, new_params, create_graph=create_graph)
        new_params = [(new_params[i] - params.inner_lr*grads[i])
                      for i in range(len(new_params))]
        del grads
    config['meta_param'] = new_params[0]
    return


def get_rl_baseline(batch, config):
    model.pick_sample('random', config)
    new_params = clone_meta_params(batch)
    inner_algo(batch, config, new_params)
    with torch.no_grad():
        output = model(batch, config)['output']
    random_baseline = batch_accuracy(output, batch)
    return random_baseline


def pick_rl_samples(batch, config):
    env_states = model.reset(batch)
    action_mask, train_mask = env_states['action_mask'], env_states['train_mask']
    for _ in range(params.n_query):
        with torch.no_grad():
            state = model.step(env_states)
        if config['mode'] == 'train':
            actions = ppo_policy.policy_old.act(state, memory, action_mask)
        else:
            with torch.no_grad():
                actions = ppo_policy.policy_old.act(state, memory, action_mask)
        action_mask[range(len(action_mask)), actions], train_mask[range(
            len(train_mask)), actions] = 0, 1
        env_states['train_mask'], env_states['action_mask'] = train_mask, action_mask
    # train_mask
    config['train_mask'] = env_states['train_mask']
    return


def run_unbiased(batch, config):
    new_params = clone_meta_params(batch)
    config['available_mask'] = batch['input_mask'].to(device).clone()
    if config['mode'] == 'train':
        random_baseline = get_rl_baseline(batch, config)
    pick_rl_samples(batch, config)
    optimizer.zero_grad()
    meta_params_optimizer.zero_grad()
    inner_algo(batch, config, new_params)
    if config['mode'] == 'train':
        res = model(batch, config)
        loss = res['loss']
        loss.backward()
        optimizer.step()
        meta_params_optimizer.step()
        ####
        final_accuracy = batch_accuracy(res['output'], batch)
        reward = final_accuracy - random_baseline
        memory.rewards.append(reward.to(device))
        ppo_policy.update(memory)
        memory.clear_memory()
        return
        #
    else:
        picked_samples = config['train_mask'].nonzero()#student_idx, q_idx
        available_samples = batch['input_mask'].to(device).clone()#nonzero()
        with torch.no_grad():
            res = model(batch, config)
        memory.clear_memory()
        return res['output'], picked_samples,available_samples


def pick_biased_samples(batch, config):
    new_params = clone_meta_params(batch)
    env_states = model.reset(batch)
    action_mask, train_mask = env_states['action_mask'], env_states['train_mask']
    for _ in range(params.n_query):
        with torch.no_grad():
            state = model.step(env_states)
            train_mask = env_states['train_mask']
        if config['mode'] == 'train':
            train_mask_sample, actions, entropy = st_policy.policy(state, action_mask)
        else:
            with torch.no_grad():
                train_mask_sample, actions, _ = st_policy.policy(
                    state, action_mask)
        action_mask[range(len(action_mask)), actions] = 0
        # env state train mask should be detached
        env_states['train_mask'], env_states['action_mask'] = train_mask + \
            train_mask_sample.data, action_mask
        if config['mode'] == 'train':
            # loss computation train mask should flow gradient
            config['train_mask'] = train_mask_sample+train_mask
            inner_algo(batch, config, new_params, create_graph=True)
            res = model(batch, config)
            loss = res['loss'] - params.lamda *entropy
            st_policy.update(loss)
    config['train_mask'] = env_states['train_mask']
    return


def run_biased(batch, config):
    new_params = clone_meta_params(batch)
    if config['mode'] == 'train':
        model.eval()
    pick_biased_samples(batch, config)
    optimizer.zero_grad()
    meta_params_optimizer.zero_grad()
    inner_algo(batch, config, new_params)
    if config['mode'] == 'train':
        model.train()
        optimizer.zero_grad()
        res = model(batch, config)
        loss = res['loss']
        loss.backward()
        optimizer.step()
        meta_params_optimizer.step()
        ####
    else:
        picked_samples = config['train_mask'].nonzero()#student_idx, q_idx
        available_samples = batch['input_mask'].to(device).clone()#nonzero()
        with torch.no_grad():
            res = model(batch, config)
        return res['output'], picked_samples, available_samples


def run_random(batch, config):
    new_params = clone_meta_params(batch)
    meta_params_optimizer.zero_grad()
    if config['mode'] == 'train':
        optimizer.zero_grad()
    ###
    config['available_mask'] = batch['input_mask'].to(device).clone()
    config['train_mask'] = torch.zeros(
        len(batch['input_mask']), params.n_question).long().to(device)

    # Random pick once
    config['meta_param'] = new_params[0]
    if sampling == 'random':
        model.pick_sample('random', config)
        inner_algo(batch, config, new_params)
    if sampling == 'active':
        for _ in range(params.n_query):
            model.pick_sample('active', config)
            inner_algo(batch, config, new_params)

    if config['mode'] == 'train':
        res = model(batch, config)
        loss = res['loss']
        loss.backward()
        optimizer.step()
        meta_params_optimizer.step()
        return
    else:
        picked_samples = config['train_mask'].nonzero()#student_idx, q_idx
        available_samples = batch['input_mask'].to(device).clone()#nonzero()
        with torch.no_grad():
            res = model(batch, config)
        output = res['output']
        return output, picked_samples, available_samples


def train_model():
    global best_val_auc, best_test_auc, best_val_score, best_test_score, best_epoch
    global run
    config['mode'] = 'train'
    config['epoch'] = epoch
    model.train()
    N = [idx for idx in range(100, 100+params.repeat)]
    for batch in train_loader:
        # Select RL Actions, save in config
        if sampling == 'unbiased':
            run_unbiased(batch, config)
        elif sampling == 'biased':
            run_biased(batch, config)
        else:
            run_random(batch, config)
        if DEBUG:
            break
    # Validation
    val_scores, val_aucs = [], []
    test_scores, test_aucs = [], []
    for idx in N:
        _, auc, acc, _ = test_model(id_=idx, split='val', get_rate=False)
        val_scores.append(acc)
        val_aucs.append(auc)
    val_score = sum(val_scores)/(len(N)+1e-20)
    val_auc = sum(val_aucs)/(len(N)+1e-20)

    if best_val_score < val_score:
        best_epoch = epoch
        best_val_score = val_score
        best_val_auc = val_auc
        # Run on test set
        for idx in N:
            _, auc, acc, exposure_rate = test_model(id_=idx, split='test',  get_rate=idx==N[-1])
            test_scores.append(acc)
            test_aucs.append(auc)
        best_test_score = sum(test_scores)/(len(N)+1e-20)
        best_test_auc = sum(test_aucs)/(len(N)+1e-20)
        if params.neptune:
            run['overlap_mu'] = exposure_rate['overlap_mu']
            run['expose_histogram'] =  exposure_rate['histogram']
            run['expose_mu'] = exposure_rate['mu']
            run['expose_chi'] = exposure_rate['chi']
            run['expose_std'] = exposure_rate['std']
    if params.neptune:
        run['Valid Accuracy'].log(val_score)
        run['Best Test Accuracy'].log(best_test_score)
        run['Best Test Auc'].log(best_test_auc)
        run['Best Valid Accuracy'].log(best_val_score)
        run['Best Valid Auc'].log(best_val_auc)
        run['Best Epoch'].log(best_epoch)
        run['Epoch'].log(epoch)
        
        
    

def test_model(id_, split='val', get_rate= False):
    if DEBUG:
        get_rate=True
    model.eval()
    config['mode'] = 'test'
    if split == 'val':
        valid_dataset.seed = id_
    elif split == 'test':
        test_dataset.seed = id_
    loader = torch.utils.data.DataLoader(
        valid_dataset if split == 'val' else test_dataset, collate_fn=collate_fn, batch_size=params.test_batch_size, num_workers=num_workers, shuffle=False, drop_last=False)
    selected, occurrence = torch.zeros(1, params.n_question).to(device), 1e-12+torch.zeros(1, params.n_question).to(device)
    random_probs = torch.zeros( params.n_question).to(device)
    exposure_rates = None

    total_loss, all_preds, all_targets = 0., [], []
    n_batch = 0
    for batch in loader:
        if sampling == 'unbiased':
            output, picked_samples, available_samples = run_unbiased(batch, config)
        elif sampling == 'biased':
            output, picked_samples, available_samples = run_biased(batch, config)
        else:
            output, picked_samples,available_samples = run_random(batch, config)
        target = batch['output_labels'].float().numpy()
        mask = batch['output_mask'].numpy() == 1
        all_preds.append(output[mask])
        all_targets.append(target[mask])
        # 
        if get_rate:
            selected.scatter_add_(1, picked_samples[:,1][None,:], torch.ones(1,picked_samples.shape[0]).to(device))
            #
            occurrence = occurrence + torch.sum(available_samples, dim =0, keepdim=True)
            temp = (params.n_query*available_samples) / torch.sum(available_samples, dim=-1, keepdim=True)
            random_probs = random_probs +  torch.sum(temp, dim=0)
        n_batch += 1
        if DEBUG:
            break
        #
    #exposure
    if get_rate:
        selected, occurrence  = selected.squeeze(0), occurrence.squeeze(0)
        exposure_rate = selected/occurrence
        mu, std = torch.mean(exposure_rate), torch.std(exposure_rate)
        histogram  = torch.histogram(exposure_rate.cpu(),bins=101, range=(-0.01+1e-6,1.+1e-6))[0].numpy().tolist()
        random_probs /= occurrence
        chi_square = torch.mean((exposure_rate -random_probs)**2. /(random_probs +1e-20))

        exposure_rates = {'mu':mu, 'std':std, 'histogram':histogram, 'chi':float(chi_square.cpu())}
        #overlap
        num =  torch.sum(selected * torch.clamp(selected-1., min=0))
        p =  torch.sum(selected)/params.n_query
        denom = params.n_query * p *(p-1.)
        exposure_rates['overlap_mu']= float(num/denom)
        

    all_pred = np.concatenate(all_preds, axis=0)
    all_target = np.concatenate(all_targets, axis=0)
    auc = compute_auc(all_target, all_pred)
    accuracy = compute_accuracy(all_target, all_pred)
    return total_loss/n_batch, auc, accuracy, exposure_rates


if __name__ == "__main__":
    params = create_parser()
    print(params)
    if DEBUG:
        params.repeat,params.gumbel = 1, True
    if params.cuda:
        if device.type!='cuda':
            from datetime import datetime
            time_now = str(datetime.utcnow())
            command = "sacct -n -j " + params.name+" --format=Jobname"
            config_name =[d for d in  (subprocess.check_output(command, shell=True)).decode("utf-8").split(' ') if '.sh' in d][0]
            line = params.name+','+params.nodes+','+str(time_now)+','+config_name +'\n'
            with open('logs/exceptions.txt', 'a') as fp:
                fp.write(line)
            raise ValueError('No Cuda Found!')

    if params.neptune:
        import neptune.new as neptune
        run = neptune.init(
            project="",  # need to add the neptune project name
            api_token="" , # need to add the neptune api token
            capture_hardware_metrics = False,
            name = params.name,
        )
        run["parameters"] = vars(params)

    config = {}
    initialize_seeds(params.seed)

    #
    base, sampling = params.model.split('-')[0], params.model.split('-')[-1]
    if base == 'biirt':
        model = MAMLModel(sampling=sampling, n_query=params.n_query,
                          n_question=params.n_question, question_dim=1).to(device)
        meta_params = [torch.Tensor(
            1, 1).normal_(-1., 1.).to(device).requires_grad_()]
    if base == 'binn':
        model = MAMLModel(sampling=sampling, n_query=params.n_query,
                          n_question=params.n_question, question_dim=params.question_dim).to(device)
        meta_params = [torch.Tensor(
            1, params.question_dim).normal_(-1., 1.).to(device).requires_grad_()]

    optimizer = torch.optim.Adam(
        model.parameters(), lr=params.lr, weight_decay=1e-8)
    meta_params_optimizer = torch.optim.SGD(
        meta_params, lr=params.meta_lr, weight_decay=2e-6, momentum=0.9)
    print(model)

    #
    if sampling == 'unbiased':
        betas = (0.9, 0.999)
        K_epochs = 4                # update policy for K epochs
        eps_clip = 0.2              # clip parameter for PPO
        memory = Memory()
        ppo_policy = PPO(params.n_question, params.n_question,
                         params.policy_lr, betas, K_epochs, eps_clip, params.lamda)
        
        print('ppo_model_summary', repr(ppo_policy.policy))
    if sampling == 'biased':
        betas = (0.9, 0.999)
        st_policy = StraightThrough(params.n_question, params.n_question,
                                    params.policy_lr, betas, params.gumbel, params.tau)
        
        print('biased_model_summary', repr(st_policy.policy))

    #
    data_path = os.path.normpath('data/train_task_'+params.dataset+'.json')
    train_data, valid_data, test_data = data_split(
        data_path, params.fold,  params.seed)
    train_dataset, valid_dataset, test_dataset = Dataset(
        train_data), Dataset(valid_data), Dataset(test_data)
    #
    num_workers = 3
    collate_fn = collate_fn(params.n_question)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, collate_fn=collate_fn, batch_size=params.train_batch_size, num_workers=num_workers, shuffle=True, drop_last=True)
    start_time = time.time()
    for epoch in range(params.n_epoch):
        train_model()
        if epoch >= (best_epoch+params.wait):
            break
