# Databricks notebook source
# MAGIC %md
# MAGIC # DAIS 2021 Data Science session: Exploration
# MAGIC 
# MAGIC Welcome to Databricks! This session will illustrate a fictional, simple, but representative day in the life of a data scientist on Databricks, who starts with data and ends up with a basic production service.
# MAGIC 
# MAGIC ## Problem: Churn
# MAGIC 
# MAGIC Imagine the case of a startup telecom company, with customers who unfortunately sometimes choose to terminate their service. It would be useful to predict when a customer might churn, to intervene. Fortunately, the company has been diligent about collecting data about customers, which might be predictive. This is new territory, the first time the company has tackled the problem. Where to start?
# MAGIC 
# MAGIC ## Data Exploration
# MAGIC 
# MAGIC We can start by simply reading the data and exploring it. There's already some useful information in the `demographic` table: customer ID, whether they have churned (or not, yet), and basic demographic information:

# COMMAND ----------

display(spark.read.table("seanowen.demographic"))

# COMMAND ----------

# MAGIC %md
# MAGIC Do the normal, predictable things. Compute summary stats. Plot some values. See what's what.

# COMMAND ----------

display(spark.read.table("seanowen.demographic").summary())

# COMMAND ----------

display(spark.read.table("seanowen.demographic"))

# COMMAND ----------

# MAGIC %md
# MAGIC This is easy, but frankly not that informative. Looking deeper would require writing some more code to explore, and, after all we do want to get on to trying out modeling too. Rather than continue, how about a coffee break while the machines do some work?
# MAGIC 
# MAGIC Use Auto ML to explore this data set and build a baseline model before continuing.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Exploration with the Feature Store
# MAGIC 
# MAGIC The model was OK, but, could probably be better with more data. There is, fortunately, more data about customers available -- additional information about the services they use, as well as some other derived, aggregated data. This was (let us say) previously used for another customer-related modeling task. This data is therefore available in the Feature Store. Why not reuse these features and see what happens, before going further?
# MAGIC 
# MAGIC Use the Feature Store to read and join everything in the `service_features` feature table.

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient, FeatureLookup

fs = FeatureStoreClient()

training_set = fs.create_training_set(spark.read.table("seanowen.demographic"), 
                                      [FeatureLookup(table_name = "seanowen.service_features", lookup_key="customerID")], 
                                      label=None, exclude_columns="customerID")

display(training_set.load_df())

# COMMAND ----------

# MAGIC %md
# MAGIC Save the augmented data set as a table, for auto ML.

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE seanowen.demographic_service

# COMMAND ----------

training_set.load_df().write.format("delta").saveAsTable("seanowen.demographic_service")

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # Model Insights with SHAP
# MAGIC 
# MAGIC As part of the exploration process, having a baseline model early helps explore the _data_ in turn. For example, the basic SHAP plots created by auto ML can be expanded to explore more of the data:

# COMMAND ----------

# MAGIC %pip install scikit-learn==1.0

# COMMAND ----------

import mlflow
import mlflow.sklearn
from shap import KernelExplainer, summary_plot
from sklearn.model_selection import train_test_split
import pandas as pd

mlflow.autolog(disable=True)

sample = spark.read.table("seanowen.demographic_service").sample(0.05).toPandas()
data = sample.drop(["Churn"], axis=1)
labels = sample["Churn"]
X_background, X_example, _, y_example = train_test_split(data, labels, train_size=0.5, random_state=42, stratify=labels)

model = mlflow.sklearn.load_model("runs:/b53bf168667a46aaadb6a06aab0bc0ac/model")

predict = lambda x: model.predict_proba(pd.DataFrame(x, columns=X_example.columns))[:,-1]
explainer = KernelExplainer(predict, X_example)
shap_values = explainer.shap_values(X=X_example)

# COMMAND ----------

summary_plot(shap_values, features=X_example)

# COMMAND ----------

# MAGIC %md
# MAGIC ... or explore how churn factors differ by gender:

# COMMAND ----------

from shap import group_difference_plot

group_difference_plot(shap_values[y_example == 1], X_example[y_example == 1]['gender'] == 'Male', feature_names=X_example.columns)

# COMMAND ----------

# MAGIC %md
# MAGIC ... or 'cluster' customers by their model explanations and see whether patterns emerge. There are clearly a group of churners that tend to be on 1-month contracts, short tenure, and equally a cluster of non-churners who are on 2-year contracts and have been long time customers. But we knew that!

# COMMAND ----------

import seaborn as sns
from sklearn.manifold import TSNE

embedded = TSNE(n_components=2, learning_rate='auto', init='pca', random_state=42).fit_transform(shap_values)

sns.set(rc = {'figure.figsize':(16,9)})
sns.scatterplot(x=embedded[:,0], y=embedded[:,1], \
                style=X_example['Contract'], \
                hue=X_example['tenure'], \
                size=(y_example == 1), size_order=[True, False], sizes=(100,200))

# COMMAND ----------


