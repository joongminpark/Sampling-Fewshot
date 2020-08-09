import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from few_shot.datasets import MiniImageNet
from few_shot.core import NShotTaskSampler, prepare_nshot_task, EvaluateFewShot, ImportanceSampler
from few_shot.matching import matching_net_episode
from few_shot.train import fit
from few_shot.callbacks import *
from few_shot.utils import setup_dirs
from config import PATH


setup_dirs()
assert torch.cuda.is_available()
device = torch.device('cuda')
torch.backends.cudnn.benchmark = True


##############
# Parameters #
##############
parser = argparse.ArgumentParser()
parser.add_argument('--dataset')
parser.add_argument('--sampling_method', type=lambda x: x.lower()[0] == 't') # Quick hack to extract boolean
parser.add_argument('--fce', type=lambda x: x.lower()[0] == 't') 
parser.add_argument('--distance', default='cosine')
parser.add_argument('--n_train', default=1, type=int)
parser.add_argument('--n_test', default=1, type=int)
parser.add_argument('--k_train', default=5, type=int)
parser.add_argument('--k_test', default=5, type=int)
parser.add_argument('--q_train', default=15, type=int)
parser.add_argument('--q_test', default=15, type=int)
parser.add_argument('--num_s_candidates', default=20, type=int)
parser.add_argument('--init_temperature', default=20.0, type=float)
parser.add_argument('--is_diversity', type=lambda x: x.lower()[0] == 't') # diversity, similarity
parser.add_argument('--lstm_layers', default=1, type=int)
parser.add_argument('--unrolling_steps', default=2, type=int)
args = parser.parse_args()


def run():
    episodes_per_epoch = 600

    if args.dataset == 'miniImageNet':
        n_epochs = 500
        dataset_class = MiniImageNet
        num_input_channels = 3
        lstm_input_size = 1600
    else:
        raise(ValueError('need to make other datasets module'))

    param_str = f'{args.dataset}_n={args.n_train}_k={args.k_train}_q={args.q_train}_' \
                f'nv={args.n_test}_kv={args.k_test}_qv={args.q_test}_' \
                f'dist={args.distance}_fce={args.fce}_sampling_method={args.sampling_method}_' \
                f'is_diversity={args.is_diversity}_epi_candidate={args.num_s_candidates}'


    #########
    # Model #
    #########
    from few_shot.models import MatchingNetwork
    model = MatchingNetwork(args.n_train, args.k_train, args.q_train, args.fce, num_input_channels,
                            lstm_layers=args.lstm_layers,
                            lstm_input_size=lstm_input_size,
                            unrolling_steps=args.unrolling_steps,
                            device=device)
    model.to(device, dtype=torch.double)


    ###################
    # Create datasets #
    ###################
    train_dataset = dataset_class('train')
    eval_dataset = dataset_class('eval')

    # Original_sampling
    if not args.sampling_method:
        train_dataset_taskloader = DataLoader(
            train_dataset,
            batch_sampler=NShotTaskSampler(train_dataset, episodes_per_epoch, args.n_train, args.k_train, args.q_train),
            num_workers=4
        )
        eval_dataset_taskloader = DataLoader(
            eval_dataset,
            batch_sampler=NShotTaskSampler(eval_dataset, episodes_per_epoch, args.n_test, args.k_test, args.q_test),
            num_workers=4
        )
    # Importance sampling
    else:
        train_dataset_taskloader = DataLoader(
            train_dataset,
            batch_sampler=ImportanceSampler(train_dataset, model,
            episodes_per_epoch, n_epochs, args.n_train, args.k_train, args.q_train,
            args.num_s_candidates, args.init_temperature, args.is_diversity),
            num_workers=4
        )
        eval_dataset_taskloader = DataLoader(
            eval_dataset,
            batch_sampler=NShotTaskSampler(eval_dataset, episodes_per_epoch, args.n_test, args.k_test, args.q_test),
            num_workers=4
        )

    ############
    # Training #
    ############
    print(f'Training Matching Network on {args.dataset}...')
    optimiser = Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.NLLLoss().cuda()


    callbacks = [
        EvaluateFewShot(
            eval_fn=matching_net_episode,
            n_shot=args.n_test,
            k_way=args.k_test,
            q_queries=args.q_test,
            taskloader=eval_dataset_taskloader,
            prepare_batch=prepare_nshot_task(args.n_test, args.k_test, args.q_test),
            fce=args.fce,
            distance=args.distance
        ),
        ModelCheckpoint(
            filepath=PATH + f'/models/matching_nets/{param_str}.pth',
            monitor=f'val_{args.n_test}-shot_{args.k_test}-way_acc',
            save_best_only=True,
        ),
        ReduceLROnPlateau(patience=20, factor=0.5, monitor=f'val_{args.n_test}-shot_{args.k_test}-way_acc'),
        CSVLogger(PATH + f'/logs/matching_nets/{param_str}.csv'),
    ]

    fit(
        model,
        optimiser,
        loss_fn,
        epochs=n_epochs,
        dataloader=train_dataset_taskloader,
        prepare_batch=prepare_nshot_task(args.n_train, args.k_train, args.q_train),
        callbacks=callbacks,
        metrics=['categorical_accuracy'],
        fit_function=matching_net_episode,
        fit_function_kwargs={'n_shot': args.n_train, 'k_way': args.k_train, 'q_queries': args.q_train, 'train': True,
                            'fce': args.fce, 'distance': args.distance}
    )

if __name__ == '__main__':
    run()