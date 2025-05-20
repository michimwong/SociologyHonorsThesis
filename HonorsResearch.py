#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 23:02:59 2025

@author: michiwong
"""

import pandas as pd
from collections import Counter, defaultdict
import re
from statsmodels.stats.contingency_tables import mcnemar
import os
from pingouin import cronbach_alpha
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the TSV file
df = pd.read_csv("/Users/michiwong/Desktop/DURF Research/DURF.tsv", sep="\t")  

### HOME COUNTRY DEMOGRAPHIC INFO
country_counts = df['What is your home country? (Where did you immigrate from?)'].value_counts()
total = country_counts.sum()

formatted = [
    f"{country} - {count} ({count / total:.2f})"
    for country, count in country_counts.items()
]

summary_country_df = pd.DataFrame({'Summary': formatted})


### SECTOR/INDUSTRY DEMOGRAPHIC INFO
# Counts the number of respondents working in each sector for a [select all that apply] question
responses = df["What sector do you work in?"].dropna()

# List of valid sectors
valid_sectors = [
    "Technology",
    "Architecture, Engineering, and Surveying",
    "Administrative Specializations",
    "Education",
    "Medicine and Health",
    "Life Sciences",
    "Mathematics and Physical Sciences",
    "Art",
    "Social Sciences",
    "Law and Jurisprudence",
    "Writing",
    "Entertainment and Recreation",
    "Museum, Library, and Archival Sciences",
    "Religion and Theology",
    "Sale Promotion",
    "Student"
]

valid_sectors_lower = [s.lower() for s in valid_sectors]

sector_counter = Counter()
other_responses = defaultdict(int)

for response in responses:
    response_lower = response.lower()
    matched = set()
    
    for i, sector in enumerate(valid_sectors_lower):
        if sector in response_lower:
            matched.add(valid_sectors[i])
            response_lower = response_lower.replace(sector, "")

    for sector in matched:
        sector_counter[sector] += 1

    # Anything left after removing known sectors is labelled as 'Other'
    leftovers = [s.strip() for s in response_lower.split(",") if s.strip()]
    for leftover in leftovers:
        sector_counter["Other"] += 1
        other_responses[leftover] += 1

total = sum(sector_counter.values())

# Create summary df with percentages
summary_data = [
    {
        "Sector": sector,
        "Count": count,
        "Percentage": round((count / total), 2)
    }
    for sector, count in sector_counter.items()
]

summary_sector_df = pd.DataFrame(summary_data).sort_values(by=["Count", "Sector"], ascending=[False, True])

sector_other_df = pd.DataFrame(other_responses.items(), columns=["Unmatched Entry", "Count"]).sort_values("Count", ascending=False)


### McNEMAR TEST OF ME VS. YOU QUALITIES OF AMERICANNESS
# Official list of Americanness qualities
official_qualities = [
    "Engages in traditional American customs (e.g., eating apple pie, watching American football)",
    "Celebrates American holidays (e.g., Independence Day, Thanksgiving)",
    "Enjoys and prepares traditional American foods (e.g., burgers, hot dogs)",
    "Displays symbols of American identity (e.g., flag, national anthem)",
    "Expresses American values (e.g., freedom of speech, equal opportunity)",
    "Feels a connection to the American 'melting pot' of cultures and ethnicities",
    "Believes in the American Dream",
    "Is familiar with with American geography and landmarks (e.g., national monuments, state capitals)",
    "Is familiar with American history and key historical figures",
    "Feels a sense of pride in American achievements and history",
    "Consumes American media (e.g., TV shows, movies, music)",
    "Understands and adheres to United States laws and regulations",
    "Participates in community events and organizations",
    "Engages in volunteer work with American communities",
    "Engages with American religious institutions or communities",
    "Votes in local, state, and national elections",
    "Is informed about United States government and political structures",
    "Follows American political discourse",
    "Follows American news and current events",
    "Uses English as a primary language in daily life",
    "Understands and uses American slang, accents, and expressions",
    "Attends American schools or universities",
    "Works in American businesses or industries",
    "Understands and participates in American economic practices (e.g., credit system, investing)",
    "Engages with American financial institutions (e.g., banks, investment firms)",
    "Owns a home in the United States",
    "Participates in the United States Social Security system",
    "Receives government assistance or subsidies",
    "Uses healthcare services and insurance provided by US healthcare systems",
    "Pays taxes in the United States",
    "Has personal or familial immigrant backgrounds",
    "Engages with immigrant communities and experiences",
    "Lives in the United States",
    "Has legal residency status in the United States",
    "Is a citizen of the United States",
    "Identifies as American",
    "Looks like an American",
    "Feels a sense of belonging in the United States",
]

def extract_qualities(raw_string, official_list):
    if pd.isna(raw_string):
        return set()
    found = set()
    for quality in official_list:
        if quality in raw_string:
            found.add(quality)
    return found

# Applies extraction to specific columns
df["yourself_set"] = df["What factors make you American?"].apply(lambda x: extract_qualities(str(x), official_qualities))
df["someone_set"] = df["What factors make someone American?"].apply(lambda x: extract_qualities(str(x), official_qualities))

# McNemar test
results = {}

for quality in official_qualities:
    # Creates binary indicators
    someone_yes = df["someone_set"].apply(lambda x: quality in x)
    yourself_yes = df["yourself_set"].apply(lambda x: quality in x)
    
    # Builds 2x2 contingency table
    yes_yes = ((someone_yes == True) & (yourself_yes == True)).sum()
    yes_no  = ((someone_yes == True) & (yourself_yes == False)).sum()
    no_yes  = ((someone_yes == False) & (yourself_yes == True)).sum()

    contingency_table = [[0, yes_no], [no_yes, 0]]

    # Performs McNemar test
    result = mcnemar(contingency_table, exact=True)
    
    results[quality] = {
        "p-value": result.pvalue,
        "Yes for Someone Else Only": yes_no,
        "Yes for Yourself Only": no_yes
    }

mcnemar_results_df = pd.DataFrame.from_dict(results, orient="index")
mcnemar_results_df = mcnemar_results_df.sort_values(by="p-value")


### LIKERT SCALE REGRESSION
likert_map = {
    "Strongly disagree": 1,
    "Disagree": 2,
    "Neither agree nor disagree": 3,
    "Agree": 4,
    "Strongly agree": 5
}

# Defines which statements are pro-immigration and which are anti
pro_immigration_cols = [
    "Please read each statement carefully and select the response that best represents your view on immigration: [Immigration has a positive impact on the United States economy]",
    "Please read each statement carefully and select the response that best represents your view on immigration: [Immigrants contribute significantly to job creation in the United States]",
    "Please read each statement carefully and select the response that best represents your view on immigration: [Workplace diversity due to immigration enhances creativity and innovation]",
    "Please read each statement carefully and select the response that best represents your view on immigration: [Companies should implement policies to support and integrate immigrant employees]",
    "Please read each statement carefully and select the response that best represents your view on immigration: [The United States government should increase funding to public services to accommodate immigrant needs]",
    "Please read each statement carefully and select the response that best represents your view on immigration: [Cultural diversity brought by immigrants enriches American society]",
    "Please read each statement carefully and select the response that best represents your view on immigration: [Immigration is essential to preserving American national identity and values]"
]

anti_immigration_cols = [
    "Please read each statement carefully and select the response that best represents your view on immigration: [Immigrants compete with native-born Americans for jobs]",
    "Please read each statement carefully and select the response that best represents your view on immigration: [Immigrants bring valuable skills and expertise to the United States workforce]",
    "Please read each statement carefully and select the response that best represents your view on immigration: [Immigration has a negative effect on wages for American workers]",
    "Please read each statement carefully and select the response that best represents your view on immigration: [Immigrants place a burden on public services such as healthcare and education]",
    "Please read each statement carefully and select the response that best represents your view on immigration: [Immigration increases demand for housing and goods, contributing to a higher cost of living]",
    "Please read each statement carefully and select the response that best represents your view on immigration: [Immigration exacerbates economic disparities within the United States]"
]

recoded_data = {}

# Convert Likert responses to numeric and reverse-code as needed
for col in pro_immigration_cols:
    recoded_data[col] = df[col].astype(str).str.strip().map(likert_map)

for col in anti_immigration_cols:
    numeric = df[col].astype(str).str.strip().map(likert_map)
    recoded_data[col] = numeric.apply(lambda x: 6 - x if pd.notna(x) else None)

attitudes_df = pd.DataFrame(recoded_data)
attitudes_df = attitudes_df.dropna().copy()

# Create composite pro-immigration score
# Result is a single score per person that reflects their overall pro-immigration attitude based on all items
attitudes_df["immigration_attitude_score"] = attitudes_df.mean(axis=1)
df["immigration_attitude_score"] = attitudes_df["immigration_attitude_score"]

# Cronbach's alpha for likert scale data
# Likert item columns (excluding the composite score)
likert_items = attitudes_df.drop(columns=["immigration_attitude_score"])
alpha, _ = cronbach_alpha(likert_items) #.79 -- Do people respond to these items in a way that suggests they’re thinking about the same general idea?


# Cleaning lenth of residency data
# Mapping length of stay text into numeric values by taking a midpoint estimate
length_mapping = {
    "<1 year": 0.5,
    "1 - 5 years": 3,    
    "5 - 10 years": 7.5,  
    "10 - 20 years": 15,  
    "20+ years": 25      
}

# Apply mapping to residency data
df["length_of_stay_num"] = df["How long have you resided in the United States?"].map(lambda x: length_mapping.get(x, None))

# Cleaning current immigration status data
#print(df["What is your immigration status now?"].unique())
# One-hot encode immigration_status with student as a reference/baseline group
status_mapping = {
    "F-1 visa": "Student/Exchange",
    "F-1 visa, OPT": "Student/Exchange",
    "OPT": "Student/Exchange",
    "J1": "Student/Exchange",
    "H1-B visa": "Worker (Temporary)",
    "L-1 visa": "Worker (Temporary)",
    "TN visa": "Worker (Temporary)",
    "EB-1 visa, H1-B visa": "Worker (Temporary)", 
    "Green card": "Permanent Resident",
    "US citizenship": "Citizen",
    "Not in the US anymore": "Other"  # We can keep or drop later depending on numbers
}
# Apply the mapping to create a new cleaned immigration status column
df["immigration_status_grouped"] = df["What is your immigration status now?"].map(lambda x: status_mapping.get(str(x).strip(), "Other"))
df = df[df["immigration_status_grouped"] != "Other"]

# Create dummies without dropping first
df_dummies = pd.get_dummies(df["immigration_status_grouped"])

# Manually drop one baseline column (e.g., Student/Exchange as reference group)
df_dummies = df_dummies.drop(columns=["Student/Exchange"], errors="ignore")

# Now build full dataset
df_encoded = pd.concat([df, df_dummies], axis=1)

immigration_status_cols = [col for col in df_encoded.columns if col.startswith("immigration_status_grouped_")]


# Regression
# Step 1: Drop rows with missing values in critical variables
df_clean = df.dropna(subset=["length_of_stay_num", "immigration_attitude_score", "immigration_status_grouped"])

# Step 2: Dummy code immigration status, manually dropping one category
# You can choose Student/Exchange as baseline group
df_dummies = pd.get_dummies(df_clean["immigration_status_grouped"])

# Drop "Student/Exchange" dummy (or whichever group you want as reference)
df_dummies = df_dummies.drop(columns=["Student/Exchange"], errors="ignore")

# Step 3: Build the final cleaned dataframe
# Combine length_of_stay_num + immigration_status dummies
X = pd.concat([df_clean[["length_of_stay_num"]], df_dummies], axis=1)
y = df_clean["immigration_attitude_score"]

# Step 4: Force types
X = X.astype(float)
y = y.astype(float)

# Step 5: Add constant (intercept)
X = sm.add_constant(X)

# Step 6: Fit regression
model = sm.OLS(y, X).fit()

# Step 7: Show results
print(model.summary())

# Step 1: Get the mean length of stay
mean_length_of_stay = df_clean["length_of_stay_num"].mean()

# Step 2: Immigration status categories
status_categories = ["Student/Exchange", "Worker (Temporary)", "Citizen", "Permanent Resident"]

# Step 3: Build prediction DataFrame
predict_df = pd.DataFrame({
    "length_of_stay_num": [mean_length_of_stay] * len(status_categories),
    "Worker (Temporary)": [1 if status == "Worker (Temporary)" else 0 for status in status_categories],
    "Citizen": [1 if status == "Citizen" else 0 for status in status_categories],
    "Permanent Resident": [1 if status == "Permanent Resident" else 0 for status in status_categories]
})

# Step 4: MANUALLY add constant column
predict_df["const"] = 1

# Step 5: Match the order exactly to model training
predict_df = predict_df[model.params.index]

# Step 6: Predict using your model
predict_df["predicted_attitude"] = model.predict(predict_df)

# Step 7: Plot
# 1. Define range of length_of_stay values (x-axis)
stay_range = np.linspace(0, 25, 100)  # from 0 to 25 years, 100 points

# 2. Immigration status groups (baseline is Student/Exchange = all dummies 0)
status_categories = ["Student/Exchange", "Citizen", "Permanent Resident", "Worker (Temporary)"]

# 3. Generate prediction data
predictions = []

for status in status_categories:
    # Create a DataFrame for each status group across the length_of_stay range
    df_pred = pd.DataFrame({
        "length_of_stay_num": stay_range,
        "Citizen": int(status == "Citizen"),
        "Permanent Resident": int(status == "Permanent Resident"),
        "Worker (Temporary)": int(status == "Worker (Temporary)")
    })

    # Add constant manually
    df_pred["const"] = 1

    # Match column order from model
    df_pred = df_pred[model.params.index]

    # Predict
    df_pred["predicted_attitude"] = model.predict(df_pred)
    df_pred["status"] = status

    predictions.append(df_pred)

# 4. Combine all predictions into one DataFrame
all_preds = pd.concat(predictions)

# 5. Plot
# Define the exact desired legend order
status_categories = ["Student/Exchange", "Worker (Temporary)", "Permanent Resident", "Citizen"]

# Plot and store handles in a dict
plt.figure(figsize=(10, 6))
line_handles = {}

for status in status_categories:
    subset = all_preds[all_preds["status"] == status]
    line, = plt.plot(
        subset["length_of_stay_num"],
        subset["predicted_attitude"],
        label=status
    )
    line_handles[status] = line  # Store handle with correct label

# Retrieve handles in exact desired order
ordered_labels = status_categories
ordered_handles = [line_handles[label] for label in ordered_labels]

# Finalize plot
plt.title("Predicted Immigration Attitudes by Length of Stay and Status")
plt.xlabel("Years in the U.S. (Length of Stay)")
plt.ylabel("Predicted Pro-Immigration Attitude Score")
plt.ylim(1, 5)
plt.grid(True)
plt.legend(ordered_handles, ordered_labels, title="Immigration Status")
plt.tight_layout()
plt.show()
