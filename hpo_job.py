"""
Distributed HPO with XGBEstimator — standalone ML Job script.
Submitted via submit_file() to avoid cloudpickle serialization issues.

NOTE: This uses XGBEstimator + XGBScalingConfig inside the Tuner train_func.
      The product team confirmed this is NOT yet a supported combination.
      The Ray autoscaler fails with: "No available node types can fulfill resource request {'CPU': 15.0}"
"""
from snowflake.ml.modeling.tune import Tuner, TunerConfig, get_tuner_context, uniform, choice, randint
from snowflake.ml.modeling.tune.search import RandomSearch
from snowflake.ml.modeling.distributors.xgboost import XGBEstimator, XGBScalingConfig
from snowflake.ml.data.data_connector import DataConnector
from snowflake.snowpark.context import get_active_session
from sklearn.metrics import roc_auc_score
import pandas as pd
import time
import sys

DB = "RRD_ML_DEMO"
SCHEMA = "DISTRIBUTED_TRAINING"
TABLE_FQN = f"{DB}.{SCHEMA}.MULTINODE_TEST_DATA"
NUM_TRIALS = 3

print("=" * 60)
print("DISTRIBUTED HPO JOB — XGBEstimator + Tuner")
print("=" * 60)

session = get_active_session()
print(f"Session: {session.get_current_role()}")

session.sql("USE WAREHOUSE ML_DEMO_WH").collect()
print("Warehouse: ML_DEMO_WH active")

# -- Load data --
print(f"\nLoading data from {TABLE_FQN}...")
df = session.table(TABLE_FQN)
feature_cols = [c for c in df.columns if c.startswith("FEAT_")]
label_col = "TARGET"
print(f"  Features: {len(feature_cols)}")
print(f"  Label: {label_col}")

train_df, test_df = df.random_split([0.8, 0.2], seed=42)

# -- Materialize splits --
# Writing to temp tables ensures the Tuner's DataConnectors don't
# need a live warehouse during trial execution (avoids silent hangs).
print("\nMaterializing train/test into temp tables...")
train_df.write.mode("overwrite").save_as_table(
    f"{DB}.{SCHEMA}.HPO_TRAIN_TEMP", table_type="temporary"
)
test_df.write.mode("overwrite").save_as_table(
    f"{DB}.{SCHEMA}.HPO_TEST_TEMP", table_type="temporary"
)
train_count = session.table(f"{DB}.{SCHEMA}.HPO_TRAIN_TEMP").count()
test_count = session.table(f"{DB}.{SCHEMA}.HPO_TEST_TEMP").count()
print(f"  Train: {train_count:,} rows")
print(f"  Test:  {test_count:,} rows")

# -- DataConnectors from materialized tables --
dataset_map = {
    "train": DataConnector.from_dataframe(session.table(f"{DB}.{SCHEMA}.HPO_TRAIN_TEMP")),
    "test": DataConnector.from_dataframe(session.table(f"{DB}.{SCHEMA}.HPO_TEST_TEMP")),
}

# -- Search space --
search_space = {
    "n_estimators": choice([50, 100, 150]),
    "max_depth": randint(4, 10),
    "learning_rate": uniform(0.01, 0.2),
}

tuner_config = TunerConfig(
    metric="auc",
    mode="max",
    search_alg=RandomSearch(random_state=42),
    num_trials=NUM_TRIALS,
    max_concurrent_trials=3,
)

# -- Capture for closure --
_feature_cols = feature_cols
_label_col = label_col


def train_func():
    """Each trial trains a distributed XGBEstimator across available workers."""
    from snowflake.ml.modeling.tune import get_tuner_context
    from snowflake.ml.modeling.distributors.xgboost import XGBEstimator, XGBScalingConfig
    from sklearn.metrics import roc_auc_score
    import pandas as pd

    ctx = get_tuner_context()
    config = ctx.get_hyper_params()
    dm = ctx.get_dataset_map()

    estimator = XGBEstimator(
        n_estimators=int(config["n_estimators"]),
        params={
            "max_depth": int(config["max_depth"]),
            "learning_rate": config["learning_rate"],
            "tree_method": "hist",
            "objective": "binary:logistic",
            "eval_metric": "auc",
        },
        scaling_config=XGBScalingConfig(num_workers=1, num_cpu_per_worker=16, use_gpu=False),
    )

    booster = estimator.fit(dm["train"], input_cols=_feature_cols, label_col=_label_col)

    predictions = estimator.predict(dm["test"])
    pred_df = predictions if isinstance(predictions, pd.DataFrame) else predictions.to_pandas()
    pred_col = [c for c in pred_df.columns if "predict" in c.lower()][0]
    target_col = _label_col if _label_col in pred_df.columns else _label_col.upper()

    auc = roc_auc_score(pred_df[target_col].astype(int), pred_df[pred_col])
    ctx.report(metrics={"auc": auc}, model=booster)


# -- Run Tuner --
print(f"\nStarting Tuner ({NUM_TRIALS} trials, 1 concurrent)...")
print(f"  Each trial uses distributed XGBEstimator (auto-detect workers/CPUs)")
sys.stdout.flush()

start = time.time()
tuner = Tuner(train_func, search_space, tuner_config)
results = tuner.run(dataset_map=dataset_map)
elapsed = time.time() - start

# -- Results --
print(f"\n{'=' * 60}")
print(f"HPO COMPLETE — {elapsed:.1f}s ({elapsed/60:.1f} min)")
print(f"{'=' * 60}")

results_df = results.results.sort_values("auc", ascending=False)
print("\nAll trials:")
print(results_df.to_string(index=False))

best = results.best_result
best_auc = float(best["auc"].iloc[0])
print(f"\nBest AUC: {best_auc:.4f}")
print(f"Best config: {best.to_dict('records')[0]}")
