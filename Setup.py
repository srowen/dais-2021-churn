# Databricks notebook source
# MAGIC %md
# MAGIC # Setup
# MAGIC 
# MAGIC This notebook contains setup code that would have been run outside of the core data science flow. These are details that aren't part of the data science demo. It's not necessarily meant to be Run All directly; these are pieces to execute as needed, for reference.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initial Data Table Setup
# MAGIC 
# MAGIC This sets up the `demographic` table, which is the initial data set considered by the data scientist. It would have been created by data engineers, in the narrative. The data set is available at https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE seanowen.demographic;

# COMMAND ----------

import pyspark.sql.functions as F

telco_df = spark.read.option("header", True).option("inferSchema", True).csv("/mnt/databricks-datasets-private/ML/telco_churn/Telco-Customer-Churn.csv")

# 0/1 -> boolean
telco_df = telco_df.withColumn("SeniorCitizen", F.col("SeniorCitizen") == 1)
# Yes/No -> boolean
for yes_no_col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
  telco_df = telco_df.withColumn(yes_no_col, F.col(yes_no_col) == "Yes")
telco_df = telco_df.withColumn("Churn", F.when(F.col("Churn") == "Yes", 1).otherwise(0))

# Contract categorical -> duration in months
telco_df = telco_df.withColumn("Contract",\
    F.when(F.col("Contract") == "Month-to-month", 1).\
    when(F.col("Contract") == "One year", 12).\
    when(F.col("Contract") == "Two year", 24))
# Empty TotalCharges -> NaN
telco_df = telco_df.withColumn("TotalCharges",\
    F.when(F.length(F.trim(F.col("TotalCharges"))) == 0, None).\
    otherwise(F.col("TotalCharges").cast('double')))

telco_df.select("customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "Churn").write.format("delta").saveAsTable("seanowen.demographic")

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Service Feature Table Setup
# MAGIC 
# MAGIC This sets up the feature store table `service_features`, which again is presumed to have been created earlier by data engineers or other teams.

# COMMAND ----------

def compute_service_features(data):
  # Count number of optional services enabled, like streaming TV
  @F.pandas_udf('int')
  def num_optional_services(*cols):
    return sum(map(lambda s: (s == "Yes").astype('int'), cols))
  
  # Below also add AvgPriceIncrease: current monthly charges compared to historical average
  service_cols = [c for c in data.columns if c not in ["gender", "SeniorCitizen", "Partner", "Dependents", "Churn"]]
  return data.select(service_cols).fillna({"TotalCharges": 0.0}).\
    withColumn("NumOptionalServices",
        num_optional_services("OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies")).\
    withColumn("AvgPriceIncrease",
        F.when(F.col("tenure") > 0, (F.col("MonthlyCharges") - (F.col("TotalCharges") / F.col("tenure")))).otherwise(0.0))

service_df = compute_service_features(telco_df)

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE seanowen.service_features;

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient()

service_features_table = fs.create_table(
  name='seanowen.service_features',
  primary_keys='customerID',
  schema=service_df.schema,
  description='Telco customer services')

# COMMAND ----------

fs.write_table("seanowen.service_features", service_df)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Webhook Setup
# MAGIC 
# MAGIC This sets up the webhook that is triggered when a new candidate model is put into Staging.

# COMMAND ----------

# MAGIC %pip install databricks-registry-webhooks

# COMMAND ----------

from databricks_registry_webhooks import RegistryWebhooksClient, JobSpec
import mlflow.tracking

host_creds = mlflow.tracking.client.MlflowClient()._tracking_client.store.get_host_creds()

job_webhook = RegistryWebhooksClient().create_webhook(
  model_name="dais-2021-churn",
  events=["MODEL_VERSION_TRANSITIONED_STAGE"],
  job_spec=JobSpec(job_id="1415341", access_token=host_creds.token),
  description="Trigger the CI/CD job when a model is moved to Staging",
  status="ACTIVE"
)

# COMMAND ----------

# MAGIC %md
# MAGIC List existing webhooks:

# COMMAND ----------

RegistryWebhooksClient().list_webhooks(model_name='dais-2021-churn')

# COMMAND ----------

# MAGIC %md
# MAGIC Delete a webhook by ID:

# COMMAND ----------

RegistryWebhooksClient().delete_webhook("6b713ba1c8b24c3a8dfa78dcce9269f6")
