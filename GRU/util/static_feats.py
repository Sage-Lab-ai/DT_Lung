# Organize the static features to use for each setup
donor_feats = ["donor_type", "donor_age", "donor_sex", "donor_bmi", "donor_tlc"]
donor_norm = ["None", "Z-score", "None", "Z-score", "Z-score"]
per_assess = ["_Pause"]
per_inter_assess = ["_Bronch", "_Recruit", "_Pause"]
hourly = [...] # selected hourly features
per_hour = [...]
image_pcs = [f"feature_pca_{f}_1h" for f in range(10)]

def get_per_assess(assess):
    return [f"{assess}{param}" for param in per_assess]

def get_per_inter_assess(assess1, assess2):
    return [f"{assess1}_{assess2}{param}" for param in per_inter_assess]

def get_hourly(hour):
    return [f"{hour}{param}" for param in hourly]

def get_per_hour(hour):
    return [f"{param}{hour}hr" for param in per_hour]

A1_A2_params = get_per_assess("A1") + get_per_inter_assess("A1", "A2") + get_hourly("70") + get_per_hour("1") + donor_feats + image_pcs
A1A2_A3_params = get_per_assess("A1") + get_per_assess("A2") + get_per_inter_assess("A1", "A2") + get_per_inter_assess("A2", "A3") + get_hourly("70") + get_hourly("80") + get_per_hour("1") + get_per_hour("2") + donor_feats + image_pcs

StaticFeaturesToInclude = {
    "A1F50_A2F50": A1_A2_params,
    "A1F50L50_A2F50": A1_A2_params,
    "N1L20A1F50L50_A2F50": A1_A2_params,
    "A1F50A2F50_A3F50": A1A2_A3_params,
    "A1F50L50A2F50_A3F50": A1A2_A3_params,
    "N1L20A1F50L50A2F50_A3F50": A1A2_A3_params,
    "A1F50_A3F50": A1_A2_params,
    "A1F50L50_A3F50": A1_A2_params,
    "N1L20A1F50L50_A3F50": A1_A2_params,
    "A1F50PA2F50_A3F50": A1_A2_params, 
    "A1F50L50PA2F50_A3F50": A1_A2_params, 
    "N1L20A1F50L50PA2F50_A3F50": A1_A2_params,
    "A1F50": None,
    "A2F50": None,
    "A3F50": None,
}

normalization_A1_A2_params = ["None"]*len(get_per_assess("A1")) + ["None"]*len(get_per_inter_assess("A1", "A2")) + ["Z-score"]*len(get_hourly("70")) + ["Z-score"]*len(get_per_hour("1")) + donor_norm + ["Z-score"]*len(image_pcs)
normalization_A1A2_A3_params = ["None"]*len(get_per_assess("A1")) + ["None"]*len(get_per_assess("A2")) + ["None"]*len(get_per_inter_assess("A1", "A2")) + ["None"]*len(get_per_inter_assess("A2", "A3")) + ["Z-score"]*len(get_hourly("70")) + ["Z-score"]*len(get_hourly("80")) + ["Z-score"]*len(get_per_hour("1")) + ["Z-score"]*len(get_per_hour("2")) + donor_norm + ["Z-score"]*len(image_pcs)

NormalizationToUse = {
    "A1F50_A2F50": normalization_A1_A2_params,
    "A1F50L50_A2F50": normalization_A1_A2_params,
    "N1L20A1F50L50_A2F50": normalization_A1_A2_params,
    "A1F50A2F50_A3F50": normalization_A1A2_A3_params,
    "A1F50L50A2F50_A3F50": normalization_A1A2_A3_params,
    "N1L20A1F50L50A2F50_A3F50": normalization_A1A2_A3_params,
    "A1F50_A3F50": normalization_A1_A2_params,
    "A1F50L50_A3F50": normalization_A1_A2_params,
    "N1L20A1F50L50_A3F50": normalization_A1_A2_params,
    "A1F50PA2F50_A3F50": normalization_A1_A2_params, 
    "A1F50L50PA2F50_A3F50": normalization_A1_A2_params, 
    "N1L20A1F50L50PA2F50_A3F50": normalization_A1_A2_params,
    "A1F50": None,
    "A2F50": None,
    "A3F50": None,
}
