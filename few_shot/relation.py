import torch
from torch.optim import Optimizer
from torch.nn import Module
from typing import Callable

from few_shot.core import create_nshot_task_label


def relation_net_episode(model: Module,
                        optimiser: Optimizer,
                        loss_fn: Callable,
                        x: torch.Tensor,
                        y: torch.Tensor,
                        n_shot: int,
                        k_way: int,
                        q_queries: int,
                        train: bool):
    """Performs a single training episode for a Relation Network.

    # Arguments
        model: Relation Network to be trained.
        optimiser: Optimiser to calculate gradient step
        loss_fn: Loss function to calculate between predictions and outputs. Should be cross-entropy
        x: Input samples of few shot classification task
        y: Input labels of few shot classification task
        n_shot: Number of examples per class in the support set
        k_way: Number of classes in the few shot classification task
        q_queries: Number of examples per class in the query set
        train: Whether (True) or not (False) to perform a parameter update

    # Returns
        loss: Loss of the Relation Network on this task
        y_pred: Predicted class probabilities for the query set on this task
    """
    if train:
        # Zero gradients
        model.train()
        optimiser.zero_grad()
    else:
        model.eval()

    # First: extracting feature (pre-relation)
    embeddings = model.encoder(x)

    # Samples are ordered by the NShotWrapper class as follows:
    # k lots of n support samples from a particular class
    # k lots of q query samples from those classes
    support = embeddings[:n_shot*k_way]
    queries = embeddings[n_shot*k_way:]                             # shape (num_queries, dim, 19, 19)

    # Element-wise sum over the samples of support sets in each class
    support_sum = compute_element_wise_sum(support, k_way, n_shot)  # shape (k_way, dim, 19, 19)

    # Concatenate channel between all queries and all support_sum
    # Output should have shape (num_query * k, output_channel * 2, 19, 19) -> (q * k * k, output_channel * 2, 19, 19)
    sup_query_pair = relation_pair(queries, support_sum, k_way, q_queries)

    # Second: extracting feature (relation)
    y_pred = model.relation(sup_query_pair).reshape(-1, k_way)  # shape (q * k, k) -> (num_query, k)

    # Create one hot label vector for the query set
    y = y.unsqueeze(-1)
    y_onehot = torch.zeros(q_queries * k_way, k_way).cuda().double()
    y_onehot = y_onehot.scatter(1, y, 1)

    # Calculate probability (y = k | x)
    loss = loss_fn(y_onehot, y_pred)

    if train:
        # Take gradient step
        loss.backward()
        optimiser.step()
    else:
        pass

    return loss, y_pred


def compute_element_wise_sum(support: torch.Tensor, k: int, n: int) -> torch.Tensor:
    """Compute element-wise sum over the embedding module outputs of all samples from each training class.

    # Arguments
        support: torch.Tensor. Tensor of shape (n * k, out_channel, 19, 19)
        k: int. "k-way" i.e. number of classes in the classification task
        n: int. "n-shot" of the classification task

    # Returns
        class_element_wise_sum: Pre-relation embedding aka sum embeddings for each class
    """
    # Reshape so the first dimension indexes by class then take the sum
    # along that dimension to generate the "element-wise sum" for each class
    class_element_wise_sum = support.reshape(k, n, -1, 19, 19).sum(dim=1)
    return class_element_wise_sum


def relation_pair(query: torch.Tensor, support: torch.Tensor, k: int, q: int) -> torch.Tensor:
    """Concatenate channel over the support of all samples from each query set.

    # Arguments
        query: torch.Tensor. Tensor of shape (n * q, output_channel, 19, 19)
        support: torch.Tensor. Tensor of shape (k, out_channel, 19, 19)
        k: int. "k-way" i.e. number of classes in the classification task
        q: int. "q-batch" of query sample in each class

    # Returns
        pairs: Cancatenate support & query along channels 
    """
    # Efficiently cancatenate each query with supports
    support_ext = support.unsqueeze(0).repeat(q * k, 1, 1, 1, 1)    # shape (num_query, k, output_channel, 19, 19)
    query_ext = query.unsqueeze(0).repeat(k, 1, 1, 1, 1)    
    query_ext = query_ext.transpose(0, 1)                           # shape (num_query, k, output_channel, 19, 19)

    # shape (num_query * k, output_channel * 2, 19, 19)
    pairs = torch.cat((support_ext, query_ext), 2).reshape(q * k * k, -1, 19, 19)
    
    return pairs
