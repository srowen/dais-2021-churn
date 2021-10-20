# Databricks notebook source
# MAGIC %md
# MAGIC # DAIS 2021 Data Science session: Automated Testing
# MAGIC 
# MAGIC This notebook is derived from the auto-generated batch inference notebook, from the MLflow Model Registry. It loads the latest Staging candidate model and, in addition to running inference on a data set, assesses model metrics on that result and from the training run. If successful, the model is promoted to Production. This is scheduled to run as a Job, triggered manually or on a schedule - or by a webhook set up to respond to state changes in the registry.
# MAGIC 
# MAGIC Load the model and set up the environment it defines:

# COMMAND ----------

from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
import os

try:
  local_path = ModelsArtifactRepository(f"models:/dais-2021-churn/staging").download_artifacts("")
except Exception:
  dbutils.notebook.exit("No staging model")

requirements_path = os.path.join(local_path, "requirements.txt")
if not os.path.exists(requirements_path):
  dbutils.fs.put(f"file:{requirements_path}", "", True)

# COMMAND ----------

# MAGIC %pip install -r $requirements_path

# COMMAND ----------

# MAGIC %md
# MAGIC Assert that the model accuracy was at least 80% at training time:

# COMMAND ----------

import mlflow.tracking

client = mlflow.tracking.MlflowClient()
latest_model_detail = client.get_latest_versions("dais-2021-churn", stages=['Staging'])[0]
accuracy = mlflow.get_run(latest_model_detail.run_id).data.metrics['training_accuracy_score']
print(f"Training accuracy: {accuracy}")
assert(accuracy >= 0.8)

# COMMAND ----------

# MAGIC %md
# MAGIC Assert that accuracy is at least 80% on a standard test set:

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient()

output_df = fs.score_batch("models:/dais-2021-churn/staging", spark.table("seanowen.demographic"))

accuracy = output_df.filter("Churn == prediction").count() / output_df.count()
print(f"Accuracy on golden test set: {accuracy}")
assert(accuracy >= 0.8)

# COMMAND ----------

# MAGIC %md
# MAGIC If successful, transition model version to Production:

# COMMAND ----------

client.transition_model_version_stage("dais-2021-churn", latest_model_detail.version, stage="Production", archive_existing_versions=True)
