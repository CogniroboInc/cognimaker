from dataclasses import dataclass, field
from typing import Type, List, Union, Optional

import pandas as pd

import lenskit.crossfold as xf
from lenskit.crossfold import SampleN, SampleFrac, LastN, LastFrac
from lenskit.topn import RecListAnalysis
from lenskit.metrics.topn import precision, recall, ndcg, recip_rank

from .evaluator import Evaluator, EvaluationResult
from ..model.recommendation import RecommendationModel, RecommendationPrediction
from ..util.const import RANDOM_SEED


@dataclass
class RecommendationEvaluator(Evaluator):
    model_cls: Type[RecommendationModel]
    num_splits: int = 5
    top_k: List[int] = field(default_factory=lambda: [1, 3, 10])
    partition_method: Union[SampleN, SampleFrac, LastN, LastFrac] = field(
        default_factory=lambda: LastN(1)
    )

    def __post_init__(self):
        # Set up metrics that require top_k
        pass

    def run(self, data: pd.DataFrame, model_params: dict) -> EvaluationResult:

        reclist = RecListAnalysis(n_jobs=1)
        reclist.add_metric(precision)
        reclist.add_metric(recall)
        reclist.add_metric(ndcg)
        reclist.add_metric(recip_rank)

        def reshape_results_at_k(result_df: pd.DataFrame, k: Optional[int] = None):
            result_df.index.name = "user"
            result_df.columns.name = "metric"
            return result_df.stack().rename("value").reset_index().assign(k=k)

        results = []
        for split in xf.partition_users(
            data,
            partitions=self.num_splits,
            method=self.partition_method,
            rng_spec=RANDOM_SEED,
        ):
            # split.train, split.test
            # split.train is the purchase data w/ all from training and
            #  non-held-out from test users
            # split.test is the held out test user rows (ground truth)
            # the held out data is dropped so if we need it we need to recover it
            model = self.model_cls(model_params)
            model.fit(split.train)
            test_users = pd.Series(split.test["user"].unique())
            prediction: RecommendationPrediction = model.predict(test_users)
            prediction = prediction.set_truth(split.test)
            n_rec_users = prediction.prediction.nunique()

            result: pd.DataFrame = reclist.compute(
                prediction.prediction, prediction.truth
            ).pipe(reshape_results_at_k).assign(n_users=n_rec_users)
            results.append(result)

            for k in self.top_k:
                pred = prediction.prediction.groupby("user").head(k)
                result: pd.DataFrame = reclist.compute(pred, prediction.truth).pipe(
                    lambda df: reshape_results_at_k(df, k)
                ).assign(n_users=n_rec_users)
                results.append(result)

        all_results = pd.concat(results).reset_index(drop=True)

        # TODO: weighted average https://stackoverflow.com/a/65204336
        agg_results = (
            all_results
            .groupby(["metric", "k"])
            .agg(value=("value", "mean"))
        )

        indicators = {}
        for (m, k) in agg_results.index:
            matk = f"{m}@{k}" if k else m
            print(f'{matk}: ({m}, {k})')
            indicators[matk] = agg_results.value.at[m, k]

        fullmodel = self.model_cls(model_params)
        fullmodel.fit(data)

        return EvaluationResult(fullmodel, indicators)
