import torch.utils.data

from gan import GAN
from metrics import Metric, unravel_metric_results
from results_storage import ResultsStorage


def evaluate_model(model_name: str, gan_model: GAN, val_dataset: torch.utils.data.Dataset,
                   metric: Metric, storage: ResultsStorage, force_rewrite: bool = False) -> None:
    exp_info = storage.get_experiment_info(model_name)
    exp_result = exp_info.get_result()

    metric_results = metric.evaluate(gan_model=gan_model, val_dataset=val_dataset)

    old_metrics = exp_result.metrics
    for metric_name, res in unravel_metric_results(metric, metric_results):
        if metric_name in old_metrics and not force_rewrite:
            continue
        exp_result.add_metric(metric_name, res)
