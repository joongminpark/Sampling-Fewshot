import logging
import argparse
import os
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Adam
from typing import Callable, List, Union
from tqdm import tqdm, trange

from few_shot.datasets import MiniImageNet
from few_shot.core import NShotTaskSampler, prepare_nshot_task, EvaluateFewShot, ImportanceSampler
from few_shot.matching import matching_net_episode
from few_shot.train import fit
from few_shot.callbacks import *
from few_shot.utils import setup_dirs, ResultWriter
from few_shot.callbacks import DefaultCallback, ProgressBarLogger, CallbackList, Callback
from few_shot.metrics import NAMED_METRICS, categorical_accuracy
from few_shot.models import MatchingNetwork

from config import PATH

logger = logging.getLogger(__name__)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str,
        default="./models/matching_nets/miniImageNet_n=1_k=5_q=15_nv=1_kv=5_qv=15_dist=cosine_fce=True_sampling_method=True_is_diversity=True_epi_candidate=20.pth", 
        help="model path")
    parser.add_argument(
        "--result_path", type=str,
        default="./results/matching_nets/miniImageNet_n=1_k=5_q=15_nv=1_kv=5_qv=15_dist=cosine_fce=True_sampling_method=True_is_diversity=True_epi_candidate=20.csv",
        help="Directory for evaluation report result (for experiments)")
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--fce', type=lambda x: x.lower()[0] == 't') 
    parser.add_argument('--distance', default='cosine')
    parser.add_argument('--n_train', default=1, type=int)
    parser.add_argument('--n_test', default=1, type=int)
    parser.add_argument('--k_train', default=5, type=int)
    parser.add_argument('--k_test', default=5, type=int)
    parser.add_argument('--q_train', default=15, type=int)
    parser.add_argument('--q_test', default=15, type=int)
    parser.add_argument('--lstm_layers', default=1, type=int)
    parser.add_argument('--unrolling_steps', default=2, type=int)
    parser.add_argument(
        "--debug", action="store_true", help="set logging level DEBUG",
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.DEBUG if args.debug else logging.INFO,
    )

    ###################
    # Create datasets #
    ###################
    episodes_per_epoch = 600

    if args.dataset == 'miniImageNet':
        n_epochs = 5
        dataset_class = MiniImageNet
        num_input_channels = 3
        lstm_input_size = 1600
    else:
        raise(ValueError('need to make other datasets module'))
    
    test_dataset = dataset_class('test')
    test_dataset_taskloader = DataLoader(
        test_dataset,
        batch_sampler=NShotTaskSampler(test_dataset, episodes_per_epoch, args.n_test, args.k_test, args.q_test),
        num_workers=4
    )

    #########
    # Model #
    #########
    model = MatchingNetwork(args.n_train, args.k_train, args.q_train, args.fce, num_input_channels,
                            lstm_layers=args.lstm_layers,
                            lstm_input_size=lstm_input_size,
                            unrolling_steps=args.unrolling_steps,
                            device=device
    ).to(device, dtype=torch.double)
    
    model.load_state_dict(torch.load(args.model_path), strict=False)
    model.eval()

    #############
    # Inference #
    #############
    logger.info("***** Epochs = %d *****", n_epochs)
    logger.info("***** Num episodes per epoch = %d *****", episodes_per_epoch)

    result_writer = ResultWriter(args.result_path)

    # just argument (function: matching_net_episode)
    prepare_batch = prepare_nshot_task(args.n_test, args.k_test, args.q_test)
    optimiser = Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.NLLLoss().cuda()

    train_iterator = trange(0, int(n_epochs), desc="Epoch",)
    for i_epoch in train_iterator:
        epoch_iterator = tqdm(test_dataset_taskloader, desc="Iteration",)
        seen = 0
        metric_name = f'test_{args.n_test}-shot_{args.k_test}-way_acc'
        metric = {metric_name: 0.0}
        for _, batch in enumerate(epoch_iterator):
            x, y = prepare_batch(batch)

            loss, y_pred = matching_net_episode(
                model,
                optimiser,
                loss_fn,
                x,
                y,
                n_shot=args.n_test,
                k_way=args.k_test,
                q_queries=args.q_test,
                train=False,
                fce=args.fce,
                distance=args.distance
            )

            seen += y_pred.shape[0]
            metric[metric_name] += categorical_accuracy(y, y_pred) * y_pred.shape[0]

        metric[metric_name] = metric[metric_name] / seen
        
        logger.info("epoch: {},     categorical_accuracy: {}".format(i_epoch, metric[metric_name]))
        result_writer.update(**metric)


if __name__ == "__main__":
    main()
