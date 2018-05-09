import matplotlib

# temporary work around down to virtualenv
# matplotlib issue.
matplotlib.use('Agg')	# 'Agg' (PNG), 'SVG', 'PDF', 'PS', must be placed before "import matplotlib.pyplot"
import numpy as np	# scientific computing library (i.e. linear algebra)
import matplotlib.pyplot as plt	# plotting library
import seaborn as sns	# statistical data visualization library

import pandas as pd	# data structures and data analysis tools library
from sklearn.linear_model import LogisticRegression

# import specific projection format.
from fairml import audit_model
from fairml import plot_dependencies

plt.style.use('ggplot')	# "plt.style.available" for a list of available styles
plt.figure(figsize=(6, 6))	# (width,height) in inches, other params available (e.g. dpi)

# read in propublica data
propublica_data = pd.read_csv("./doc/example_notebooks/"
							  "propublica_data_for_fairml.csv")	# returns DataFrame object

# quick data processing
compas_rating = propublica_data.score_factor.values	# score_factor is a feature in "propublica_data_for_fairml.csv"
													# features can be accessed as direct attributes of the object
													# compas_rating are the COMPAS results (recidivism prediction)
propublica_data = propublica_data.drop("score_factor", 1)	# remove column (argument 1), with label "score_factor"

#  quick setup of Logistic regression
#  perhaps use a more crazy classifier
clf = LogisticRegression(penalty='l2', C=0.01)	# C = regularization strength, smaller values specify stronger regularization
clf.fit(propublica_data.values, compas_rating)	# fit(training_vector, target_vector)
												# approximates the real predictive model based on its input and output data

#  call audit model
importancies, _ = audit_model(clf.predict, propublica_data)

# print feature importance
print(importancies)

# generate feature dependence plot
fig = plot_dependencies(
	importancies.median(),
	reverse_values=False,
	title="FairML feature dependence logistic regression model"
)

file_name = "fairml_propublica_linear_direct.png"
plt.savefig(file_name, transparent=False, bbox_inches='tight', dpi=250)
