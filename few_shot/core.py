from torch.utils.data import Sampler
from typing import List, Iterable, Callable, Tuple
from torch.nn import Module
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import torch.nn.functional as F

from few_shot.utils import softmax_with_temperature
from few_shot.metrics import categorical_accuracy
from few_shot.callbacks import Callback
from config import EPSILON


# Original meta-sampling (Random sampling -> In one support set, all case is considered)
# My Idea: Sampling some support sets -> sampling with prob given Importance: diversity + similarity
class ImportanceSampler(Sampler):
    def __init__(self,
                 dataset: torch.utils.data.Dataset,
                 model: Module,
                 episodes_per_epoch: int = None,
                 total_epochs: int = None,
                 n: int = None,
                 k: int = None,
                 q: int = None,
                 num_s_candidates = None,
                 init_temperature = None,
                 is_diversity = True,
                 num_tasks: int = 1,
                 fixed_tasks: List[Iterable[int]] = None):
        """PyTorch Sampler subclass that generates batches of n-shot, k-way, q-query tasks.

        Each n-shot task contains a "support set" of `k` sets of `n` samples and a "query set" of `k` sets
        of `q` samples. The support set and the query set are all grouped into one Tensor such that the first n * k
        samples are from the support set while the remaining q * k samples are from the query set.

        The support and query sets are sampled such that they are disjoint i.e. do not contain overlapping samples.

        # Arguments
            dataset: Instance of torch.utils.data.Dataset from which to draw train samples
            episodes_per_epoch: Arbitrary number of batches of n-shot tasks to generate in one epoch
            total_epochs: Arbitrary number of epochs to train model
            model: Model to be used for manifold space
            n_shot: int. Number of samples for each class in the n-shot classification tasks.
            k_way: int. Number of classes in the n-shot classification tasks.
            q_queries: int. Number query samples for each class in the n-shot classification tasks.
            num_s_candidates: Number support set candidates to sample batch
            init_temperature: Number initial temperature for softmax with temperature (after iter//2 -> 1)
            is_diversity: Whether metric measure is diversity or similarity
            num_tasks: Number of n-shot tasks to group into a single batch
            fixed_tasks: If this argument is specified this Sampler will always generate tasks from
                the specified classes
        """
        super(ImportanceSampler, self).__init__(dataset)
        self.episodes_per_epoch = episodes_per_epoch
        self.total_epochs = total_epochs
        self.dataset = dataset
        self.model = model
        if num_tasks < 1:
            raise ValueError('num_tasks must be >= 1.')

        self.num_tasks = num_tasks
        # TODO: Raise errors if initialise badly
        self.k = k
        self.n = n
        self.q = q
        self.num_s_candidates = num_s_candidates
        self.init_temperature = init_temperature
        self.is_diversity = is_diversity
        # num_sample: Number of dataset sample to use diversity measure (support set)
        self.num_sample = n    # 1, 3, 5...
        self.fixed_tasks = fixed_tasks

        self.i_task = 0
        self.i_iter= 0

    def __len__(self):
        return self.episodes_per_epoch

    def __iter__(self):
        for epi in range(self.episodes_per_epoch):
            batch = []

            for task in range(self.num_tasks):
                if self.fixed_tasks is None:
                    # Get random classes with (num_s_candidates) number support sets
                    # support_candidates: List[Iterable[num_s_candidates]]  [[num_sample * k],...,[num_sample * k]]
                    support_candidates = []
                    episode_classes_set = []
                    for i in range(self.num_s_candidates):
                        episode_classes = list(np.random.choice(self.dataset.df['class_id'].unique(), 
                        size=self.k, replace=False))
                        episode_classes_set.append(episode_classes)
                        
                        # [1(num_sample), 2(num_sample), ... , k(num_sample)]: len <num_sample * k>
                        # TODO: each num_sample -> need to average
                        episode_id = []
                        for j in episode_classes:
                            tmp_df = self.dataset.df[self.dataset.df['class_id'] == j].sample(self.num_sample)
                            episode_id.extend(tmp_df['id'])
                        support_candidates.append(episode_id)
                else:
                    raise(ValueError('cant do fixed_tasks with importance sampling.'))
                
                self.model.eval()
                diversity = []
                # diversity: List[Iterable[num_s_candidates]]
                for i in range(self.num_s_candidates):
                    # get train-support sets
                    # (1) "num_sample" number of samples for each label in train dataset (support set)
                    support_id = support_candidates[i]
                    support_train_items = list(map(self.dataset.__getitem__, support_id))
                    support_train_features = list(zip(*support_train_items))[0]
                    support_train_features = torch.stack(support_train_features)
                    support_train_features = support_train_features.double().cuda()

                    train_embeddings = self.model.encoder(support_train_features)      # shape: (num_sample*k, 1600)
                    
                    # (2) get feature for each label (train support set) using mean vector by manifold space
                    # DO: Each num_sample -> average
                    average_embeddings = []
                    for i in range(self.k):
                        if self.num_sample == 1:
                            tmp = torch.mean(train_embeddings[self.num_sample*i:self.num_sample*(i+1)], dim=0)
                        else:
                            tmp = train_embeddings[i]
                        average_embeddings.append(tmp)

                    average_embeddings = torch.stack(average_embeddings)    # shape: (k, 1600)
                    
                    norm_embedding = average_embeddings / (average_embeddings.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)
                    trans_dot_norm = torch.mm(norm_embedding, norm_embedding.transpose(0,1)) # shape: (k-1, k-1)
                    determinant = torch.det(trans_dot_norm).item()
                    determinant = determinant**0.5 # square root
                    diversity.append(determinant)
                
                diversity = np.array(diversity)
                diversity[np.isnan(diversity)] = EPSILON

                '''
                # applied to softmax with temperature for a half of entire iterations (T: 20 -> 1)
                self.i_iter += 1
                
                temp_change_iteration = (self.total_epochs*self.episodes_per_epoch) // 3
                if temp_change_iteration > self.i_iter:
                    temperature = self.init_temperature - (self.init_temperature - 1)*(self.i_iter/temp_change_iteration)
                else:
                    temperature = 1

                if self.is_diversity:
                    supports_sampling_rate = softmax_with_temperature(diversity, temperature)
                else:
                    # similarity
                    supports_sampling_rate = softmax_with_temperature(-diversity, temperature)
                '''

                if self.is_diversity:
                    supports_sampling_rate = softmax_with_temperature(diversity, 1)
                else:
                    similarity = (-1 * diversity)
                    supports_sampling_rate = softmax_with_temperature(similarity, 1)

                
                support_choice = np.random.choice(
                    list(range(self.num_s_candidates)),
                    size=1, replace=False,
                    p=supports_sampling_rate
                )

                support_candidates_id = support_candidates[support_choice[0]]
                sampling_support_classes = episode_classes_set[support_choice[0]]

                df = self.dataset.df[self.dataset.df['class_id'].isin(sampling_support_classes)]
                support_k = {k: None for k in sampling_support_classes}
                
                # Select support examples
                batch.extend(support_candidates_id)

                # self.num_sample == self.n
                for k in sampling_support_classes:
                    support_k[k] = support_candidates_id[k*self.num_sample : (k+1)*self.num_sample]
                
                # Select Query examples
                for k in sampling_support_classes:
                    query = df[(df['class_id'] == k) & (~df['id'].isin(support_k[k]))].sample(self.q)
                    for i, q in query.iterrows():
                        batch.append(q['id'])

            yield np.stack(batch)


class NShotTaskSampler(Sampler):
    def __init__(self,
                 dataset: torch.utils.data.Dataset,
                 episodes_per_epoch: int = None,
                 n: int = None,
                 k: int = None,
                 q: int = None,
                 num_tasks: int = 1,
                 fixed_tasks: List[Iterable[int]] = None):
        """PyTorch Sampler subclass that generates batches of n-shot, k-way, q-query tasks.

        Each n-shot task contains a "support set" of `k` sets of `n` samples and a "query set" of `k` sets
        of `q` samples. The support set and the query set are all grouped into one Tensor such that the first n * k
        samples are from the support set while the remaining q * k samples are from the query set.

        The support and query sets are sampled such that they are disjoint i.e. do not contain overlapping samples.

        # Arguments
            dataset: Instance of torch.utils.data.Dataset from which to draw samples
            episodes_per_epoch: Arbitrary number of batches of n-shot tasks to generate in one epoch
            n_shot: int. Number of samples for each class in the n-shot classification tasks.
            k_way: int. Number of classes in the n-shot classification tasks.
            q_queries: int. Number query samples for each class in the n-shot classification tasks.
            num_tasks: Number of n-shot tasks to group into a single batch
            fixed_tasks: If this argument is specified this Sampler will always generate tasks from
                the specified classes
        """
        super(NShotTaskSampler, self).__init__(dataset)
        self.episodes_per_epoch = episodes_per_epoch
        self.dataset = dataset
        if num_tasks < 1:
            raise ValueError('num_tasks must be > 1.')

        self.num_tasks = num_tasks
        # TODO: Raise errors if initialise badly
        self.k = k
        self.n = n
        self.q = q
        self.fixed_tasks = fixed_tasks

        self.i_task = 0

    def __len__(self):
        return self.episodes_per_epoch

    def __iter__(self):
        for _ in range(self.episodes_per_epoch):
            batch = []

            for task in range(self.num_tasks):
                if self.fixed_tasks is None:
                    # Get random classes
                    episode_classes = np.random.choice(self.dataset.df['class_id'].unique(), size=self.k, replace=False)
                else:
                    # Loop through classes in fixed_tasks
                    episode_classes = self.fixed_tasks[self.i_task % len(self.fixed_tasks)]
                    self.i_task += 1

                df = self.dataset.df[self.dataset.df['class_id'].isin(episode_classes)]

                support_k = {k: None for k in episode_classes}
                for k in episode_classes:
                    # Select support examples
                    support = df[df['class_id'] == k].sample(self.n)
                    support_k[k] = support

                    for i, s in support.iterrows():
                        batch.append(s['id'])

                for k in episode_classes:
                    query = df[(df['class_id'] == k) & (~df['id'].isin(support_k[k]['id']))].sample(self.q)
                    for i, q in query.iterrows():
                        batch.append(q['id'])

            yield np.stack(batch)


class EvaluateFewShot(Callback):
    """Evaluate a network on  an n-shot, k-way classification tasks after every epoch.

    # Arguments
        eval_fn: Callable to perform few-shot classification. Examples include `proto_net_episode`,
            `matching_net_episode` and `meta_gradient_step` (MAML).
        n_shot: int. Number of samples for each class in the n-shot classification tasks.
        k_way: int. Number of classes in the n-shot classification tasks.
        q_queries: int. Number query samples for each class in the n-shot classification tasks.
        task_loader: Instance of NShotWrapper class
        prepare_batch: function. The preprocessing function to apply to samples from the dataset.
        prefix: str. Prefix to identify dataset.
    """

    def __init__(self,
                 eval_fn: Callable,
                 n_shot: int,
                 k_way: int,
                 q_queries: int,
                 taskloader: torch.utils.data.DataLoader,
                 prepare_batch: Callable,
                 prefix: str = 'val_',
                 **kwargs):
        super(EvaluateFewShot, self).__init__()
        self.eval_fn = eval_fn
        self.n_shot = n_shot
        self.k_way = k_way
        self.q_queries = q_queries
        self.taskloader = taskloader
        self.prepare_batch = prepare_batch
        self.prefix = prefix
        self.kwargs = kwargs
        self.metric_name = f'{self.prefix}{self.n_shot}-shot_{self.k_way}-way_acc'

    def on_train_begin(self, logs=None):
        self.loss_fn = self.params['loss_fn']
        self.optimiser = self.params['optimiser']

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        seen = 0
        totals = {'loss': 0, self.metric_name: 0}
        for batch_index, batch in enumerate(self.taskloader):
            x, y = self.prepare_batch(batch)

            loss, y_pred = self.eval_fn(
                self.model,
                self.optimiser,
                self.loss_fn,
                x,
                y,
                n_shot=self.n_shot,
                k_way=self.k_way,
                q_queries=self.q_queries,
                train=False,
                **self.kwargs
            )

            seen += y_pred.shape[0]

            totals['loss'] += loss.item() * y_pred.shape[0]
            totals[self.metric_name] += categorical_accuracy(y, y_pred) * y_pred.shape[0]

        logs[self.prefix + 'loss'] = totals['loss'] / seen
        logs[self.metric_name] = totals[self.metric_name] / seen


def prepare_nshot_task(n: int, k: int, q: int) -> Callable:
    """Typical n-shot task preprocessing.

    # Arguments
        n: Number of samples for each class in the n-shot classification task
        k: Number of classes in the n-shot classification task
        q: Number of query samples for each class in the n-shot classification task

    # Returns
        prepare_nshot_task_: A Callable that processes a few shot tasks with specified n, k and q
    """
    def prepare_nshot_task_(batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create 0-k label and move to GPU.

        TODO: Move to arbitrary device
        """
        x, y = batch
        x = x.double().cuda()
        # Create dummy 0-(num_classes - 1) label
        y = create_nshot_task_label(k, q).cuda()
        return x, y

    return prepare_nshot_task_


def create_nshot_task_label(k: int, q: int) -> torch.Tensor:
    """Creates an n-shot task label.

    Label has the structure:
        [0]*q + [1]*q + ... + [k-1]*q

    # TODO: Test this

    # Arguments
        k: Number of classes in the n-shot classification task
        q: Number of query samples for each class in the n-shot classification task

    # Returns
        y: Label vector for n-shot task of shape [q * k, ]
    """
    y = torch.arange(0, k, 1 / q).long()
    return y
