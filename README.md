# CrimeDatasetVisualization
Crime Data Analysis &amp; Prediction using Machine Learning  -Crime Data Analysis project using EDA &amp; ML to study victim profiles, crime trends, hotspots, and risk prediction with Random Forest, Decision Trees, clustering, and visual insights.

CODE:-
#The objective of this analysis is to predict the age of a crime victim based on 
#the time of occurrence and geographical location (latitude and longitude) of the crime using a supervised machine learning regression model.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv(r"C:\Users\prach\Downloads\Crime_Data_from_2020_to_Present.csv")

age_df = df[["Vict Age", "TIME OCC", "LAT", "LON"]].dropna()

age_features = age_df[["TIME OCC", "LAT", "LON"]]
age_target = age_df["Vict Age"]

age_scaler = StandardScaler()
age_features_scaled = age_scaler.fit_transform(age_features)

age_X_train, age_X_test, age_y_train, age_y_test = train_test_split(
    age_features_scaled, age_target, test_size=0.2, random_state=42
)

age_model = LinearRegression()
age_model.fit(age_X_train, age_y_train)

age_predictions = age_model.predict(age_X_test)

print("Victim Age RMSE:", np.sqrt(mean_squared_error(age_y_test, age_predictions)))
print("Victim Age R2:", r2_score(age_y_test, age_predictions))

plt.scatter(age_y_test, age_predictions, alpha=0.5)
plt.xlabel("Actual Vict Age")
plt.ylabel("Predicted Vict Age")
plt.title("Victim Age Prediction")
plt.show()
#The objective of this analysis is to classify and predict the type of crime committed (limited to the top five most frequent crime categories) based
#on victim characteristics and incident-related
#factors such as premises description, weapon used, and victim sex using a Random Forest classification model.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report


crime_df2 = pd.read_csv(r"C:\Users\prach\Downloads\Crime_Data_from_2020_to_Present.csv")

# Keep only required columns
crime_df2 = crime_df2[
    ["Crm Cd Desc", "Premis Desc", "Weapon Desc", "Vict Sex"]
].dropna()

# TOP 5 crime types 
top_crimes = crime_df2["Crm Cd Desc"].value_counts().nlargest(5).index
crime_df2 = crime_df2[crime_df2["Crm Cd Desc"].isin(top_crimes)]

# Features & Target
crime_features2 = crime_df2.drop("Crm Cd Desc", axis=1)
crime_target2 = crime_df2["Crm Cd Desc"]

# Preprocessing
crime_encoder2 = ColumnTransformer([
    ("cat_encoder", OneHotEncoder(handle_unknown="ignore"), crime_features2.columns)
])

# Model
crime_model2 = Pipeline([
    ("preprocess", crime_encoder2),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train-test split
X_train2, X_test2, y_train2, y_test2 = train_test_split(
    crime_features2, crime_target2, test_size=0.2, random_state=42
)

# Train model
crime_model2.fit(X_train2, y_train2)

# Predictions
crime_predictions2 = crime_model2.predict(X_test2)

#OUTPUT (THIS WILL PRINT)
print("Classification Report:\n")
print(classification_report(y_test2, crime_predictions2))

# Confusion Matrix Plot
crime_cm2 = confusion_matrix(y_test2, crime_predictions2)

plt.figure(figsize=(8,6))
sns.heatmap(
    crime_cm2,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=top_crimes,
    yticklabels=top_crimes
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Crime Type Classification (Top 5 Crimes)")
plt.show()
#The objective of this model is to predict a crime risk score based on temporal and situational factors such as time of occurrence,
#presence of a weapon, whether the crime occurred at night, and whether the location was a public place using a Random Forest regression approach.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
risk_df = pd.read_csv(r"C:\Users\prach\Downloads\Crime_Data_from_2020_to_Present.csv")

# Select required columns
risk_df = risk_df[["TIME OCC", "Weapon Desc", "Premis Desc"]].dropna()


# Create weapon_flag
weapon_flag_list = []

for weapon in risk_df["Weapon Desc"]:
    if "NONE" in str(weapon).upper():
        weapon_flag_list.append(0)
    else:
        weapon_flag_list.append(1)

risk_df["weapon_flag"] = weapon_flag_list


# Create night_flag 

night_flag_list = []

for time in risk_df["TIME OCC"]:
    if time >= 1900 or time <= 600:
        night_flag_list.append(1)
    else:
        night_flag_list.append(0)

risk_df["night_flag"] = night_flag_list


# Create public_flag (0/1)

public_flag_list = []

for premis in risk_df["Premis Desc"]:
    premis_text = str(premis).upper()
    if "STREET" in premis_text or "PARK" in premis_text or "BUS" in premis_text:
        public_flag_list.append(1)
    else:
        public_flag_list.append(0)

risk_df["public_flag"] = public_flag_list

# Create crime risk score

risk_df["crime_risk_score"] = (
    risk_df["weapon_flag"] +
    risk_df["night_flag"] +
    risk_df["public_flag"]
)


# Features and target

risk_features = risk_df[
    ["TIME OCC", "weapon_flag", "night_flag", "public_flag"]
]
risk_target = risk_df["crime_risk_score"]


# Train-test split

risk_X_train, risk_X_test, risk_y_train, risk_y_test = train_test_split(
    risk_features,
    risk_target,
    test_size=0.2,
    random_state=42
)


# Train model

risk_model = RandomForestRegressor(n_estimators=150, random_state=42)
risk_model.fit(risk_X_train, risk_y_train)


# Predictions

risk_predictions = risk_model.predict(risk_X_test)


# Evaluation

print("Crime Risk RMSE:", np.sqrt(mean_squared_error(risk_y_test, risk_predictions)))
print("Crime Risk R2:", r2_score(risk_y_test, risk_predictions))


# Plot-Actual vs Predicted

plt.figure(figsize=(6,6))
plt.scatter(risk_y_test, risk_predictions, alpha=0.5)
plt.xlabel("Actual Crime Risk")
plt.ylabel("Predicted Crime Risk")
plt.title("Crime Risk Prediction (Random Forest)")
plt.show()
#The objective of this code is to identify and visualize crime hotspots by grouping crime incidents based on their geographical coordinates
#(latitude and longitude) using K-Means clustering.
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

hotspot_df = pd.read_csv(r"C:\Users\prach\Downloads\Crime_Data_from_2020_to_Present.csv")

hotspot_df = hotspot_df[["LAT", "LON"]].dropna()

hotspot_model = KMeans(n_clusters=5, random_state=42)
hotspot_df["hotspot_label"] = hotspot_model.fit_predict(hotspot_df)

plt.scatter(
    hotspot_df["LON"],
    hotspot_df["LAT"],
    c=hotspot_df["hotspot_label"],
    cmap="tab10",
    s=5
)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Crime Hotspots")
plt.show()
#The objective of this analysis is to identify and visualize crime hotspot intensity by clustering crime incidents based on both spatial 
#(latitude and longitude) and temporal (time of occurrence)
#characteristics using K-Means clustering.
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


hotspot_time_df = pd.read_csv(r"C:\Users\prach\Downloads\Crime_Data_from_2020_to_Present.csv")

# Select required columns
hotspot_time_df = hotspot_time_df[["LAT", "LON", "TIME OCC"]].dropna()//100

# Convert time to hour
hotspot_time_df["hour"] = hotspot_time_df["TIME OCC"] 

# Features for clustering
hotspot_features = hotspot_time_df[["LAT", "LON", "hour"]]

# Scaling important for clustering
hotspot_scaler = StandardScaler()
hotspot_scaled = hotspot_scaler.fit_transform(hotspot_features)

# KMeans
hotspot_kmeans = KMeans(n_clusters=5, random_state=42)
hotspot_time_df["intensity_cluster"] = hotspot_kmeans.fit_predict(hotspot_scaled)

# Plot crime hotspot intensity
plt.figure(figsize=(8,6))
plt.scatter(
    hotspot_time_df["LON"],
    hotspot_time_df["LAT"],
    c=hotspot_time_df["intensity_cluster"],
    cmap="viridis",
    s=8
)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Crime Hotspot Intensity Clustering")
plt.colorbar(label="Intensity Cluster")
plt.show()

# The objective of this analysis is to study the variation in crime intensity across different hours of the day by visualizing the number of crimes 
#occurring at each hour using a line chart.
hourly_crime_count = hotspot_time_df.groupby("hour").size()

plt.figure(figsize=(8,5))
plt.plot(
    hourly_crime_count.index,
    hourly_crime_count.values,
    marker='o'
)
plt.xlabel("Hour of Day")
plt.ylabel("Number of Crimes")
plt.title("Crime Intensity Trend Over Time")
plt.grid(True)
plt.show()
#The objective of this analysis is to examine the distribution of victim age and gender across major crime types in order to identify 
#demographic patterns and understand how different crimes affect
#different sections of the population.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


victim_df = pd.read_csv(r"C:\Users\prach\Downloads\Crime_Data_from_2020_to_Present.csv")

# Select required columns
victim_df = victim_df[[
    "Vict Age",
    "Vict Sex",
    "Crm Cd Desc"
]].dropna()

# Filter realistic ages
victim_df = victim_df[
    (victim_df["Vict Age"] > 0) &
    (victim_df["Vict Age"] < 100)
]

# top 5 crime types 
top_victim_crimes = victim_df["Crm Cd Desc"].value_counts().nlargest(5).index
victim_df = victim_df[victim_df["Crm Cd Desc"].isin(top_victim_crimes)]


# PLOT 1- Box Plot (Age vs Crime)

plt.figure(figsize=(10,6))
sns.boxplot(
    data=victim_df,
    x="Crm Cd Desc",
    y="Vict Age"
)
plt.xticks(rotation=45)
plt.xlabel("Crime Type")
plt.ylabel("Victim Age")
plt.title("Victim Age Distribution Across Crime Types")
plt.show()


# PLOT 2- Bar Chart (Gender vs Crime)

gender_crime_counts = (
    victim_df
    .groupby(["Crm Cd Desc", "Vict Sex"])
    .size()
    .reset_index(name="count")
)

plt.figure(figsize=(10,6))
sns.barplot(
    data=gender_crime_counts,
    x="Crm Cd Desc",
    y="count",
    hue="Vict Sex"
)
plt.xticks(rotation=45)
plt.xlabel("Crime Type")
plt.ylabel("Number of Victims")
plt.title("Gender Distribution Across Crime Types")
plt.legend(title="Victim Sex")
plt.show()
#The objective of this analysis is to examine how the average age of crime victims varies across different times of occurrence, 
#in order to identify time periods
#associated with younger or older victim groups.
import pandas as pd
import matplotlib.pyplot as plt

trend_df = pd.read_csv(r"C:\Users\prach\Downloads\Crime_Data_from_2020_to_Present.csv")

trend_df = trend_df[["Vict Age", "TIME OCC"]].dropna()

age_time_group = trend_df.groupby("TIME OCC")["Vict Age"].mean()

plt.plot(age_time_group.index, age_time_group.values)
plt.xlabel("Time of Occurrence")
plt.ylabel("Average Victim Age")
plt.title("Average Victim Age Trend by Time")
plt.show()
