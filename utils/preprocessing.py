import pandas as pd
import numpy as np
import joblib

def load_artifacts():
    aft = joblib.load("model/weibull_aft_model.pkl")
    encoder = joblib.load("model/encoder.pkl")
    feature_names = joblib.load("model/feature_names.pkl")
    return aft, encoder, feature_names

def preprocess_input(
    species,
    soil,
    light,
    season,
    census,
    emf,
    myco_type,
    encoder,
    feature_names
):
    if emf < 0:
        raise ValueError("EMF must be non-negative")

    if census <= 0:
        raise ValueError("Census must be positive")

    row = {
        "Species": species,
        "Light_Cat": light,
        "Soil": soil,
        "Sterile": "Sterile",
        "Conspecific": "Heterospecific",
        "Myco_type": myco_type,
        "PlantSeason": season,
        "Census": census,
        "EMF_log": np.log1p(emf)
    }

    df = pd.DataFrame([row])

    missing = set(encoder.feature_names_in_) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    df = df[encoder.feature_names_in_]

    encoded = encoder.transform(df)

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out()
    )

    encoded_df = encoded_df.reindex(
        columns=feature_names,
        fill_value=0
    )

    return encoded_df
