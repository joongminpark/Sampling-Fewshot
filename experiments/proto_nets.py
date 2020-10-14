from torch.optim import Adam
from torch.utils.data import DataLoader
import argparse

from few_shot.datasets import MiniImageNet
from few_shot.models import get_few_shot_encoder
from few_shot.core import NShotTaskSampler, EvaluateFewShot, prepare_nshot_task, ImportanceSampler
from few_shot.proto import proto_net_episode
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
parser.add_argument('--distance', default='l2')
parser.add_argument('--n_train', default=1, type=int)
parser.add_argument('--n_test', default=1, type=int)
parser.add_argument('--k_train', default=5, type=int)
parser.add_argument('--k_test', default=5, type=int)
parser.add_argument('--q_train', default=15, type=int)
parser.add_argument('--q_test', default=15, type=int)
parser.add_argument('--num_s_candidates', default=20, type=int)
parser.add_argument('--init_temperature', default=20.0, type=float)
parser.add_argument('--is_diversity', type=lambda x: x.lower()[0] == 't') # diversity, similarity
args = parser.parse_args()



def run():
    episodes_per_epoch = 600

    '''
    ###### LearningRateScheduler ######
    drop_lr_every = 20
    def lr_schedule(epoch, lr):
        # Drop lr every 2000 episodes
        if epoch % drop_lr_every == 0:
            return lr / 2
        else:
            return lr
    # callbacks add: LearningRateScheduler(schedule=lr_schedule)
    '''

    if args.dataset == 'miniImageNet':
        n_epochs = 500
        dataset_class = MiniImageNet
        num_input_channels = 3
    else:
        raise(ValueError('need to make other datasets module'))


    param_str = f'{args.dataset}_nt={args.n_train}_kt={args.k_train}_qt={args.q_train}_' \
                f'nv={args.n_test}_kv={args.k_test}_qv={args.q_test}_' \
                f'dist={args.distance}_sampling_method={args.sampling_method}_is_diverisity={args.is_diversity}'

    print(param_str)

    #########
    # Model #
    #########
    model = get_few_shot_encoder(num_input_channels)
    model.to(device, dtype=torch.double)

    ###################
    # Create datasets #
    ###################
    train_dataset = dataset_class('train')
    eval_dataset = dataset_class('eval')

    # Original sampling
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
        # ImportanceSampler: Latent space of model
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
    print(f'Training Prototypical network on {args.dataset}...')
    optimiser = Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.NLLLoss().cuda()


    callbacks = [
        EvaluateFewShot(
            eval_fn=proto_net_episode,
            n_shot=args.n_test,
            k_way=args.k_test,
            q_queries=args.q_test,
            taskloader=eval_dataset_taskloader,
            prepare_batch=prepare_nshot_task(args.n_test, args.k_test, args.q_test),
            distance=args.distance
        ),
        ModelCheckpoint(
            filepath=PATH + f'/models/proto_nets/{param_str}.pth',
            monitor=f'val_{args.n_test}-shot_{args.k_test}-way_acc',
            save_best_only=True,
        ),
        ReduceLROnPlateau(patience=40, factor=0.5, monitor=f'val_{args.n_test}-shot_{args.k_test}-way_acc'),
        CSVLogger(PATH + f'/logs/proto_nets/{param_str}.csv'),
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
        fit_function=proto_net_episode,
        fit_function_kwargs={'n_shot': args.n_train, 'k_way': args.k_train, 'q_queries': args.q_train, 'train': True,
                            'distance': args.distance},
    )

if __name__ == '__main__':
    run()