from schnetpack.train import Metric

class MultitaskMetricWrapper(Metric):
    """
    This class expects batches with the columns validity and value.
    Unly the valid network estimats and target values are passed to the wrapped
    metric.
    """
    def __init__(self, metric: Metric):
        self.metric = metric
        self.name = metric.name

    def add_batch(self, batch, result):
        target = self.metric.target
        valid = batch[target][:, 0] > 0
        batch[target] = batch[target][valid, 1]
        model_output = self.metric.model_output
        if model_output is None:
            result = result[valid]
        else:
            result[model_output] = result[model_output][valid]
        self.metric.add_batch(batch, result)

    def aggregate(self):
        """Aggregate metric over all previously added batches."""
        return self.metric.aggregate()

    def reset(self):
        """Reset metric attributes after aggregation to collect new batches."""
        self.metric.reset()
