def month_to_season(month: int) -> str:
    if month in [6, 7, 8, 9]:
        return "Monsoon"
    elif month in [10, 11]:
        return "PostMonsoon"
    elif month in [12, 1, 2]:
        return "Winter"
    else:
        return "Summer"


def risk_label(prob: float) -> str:
    if prob >= 0.75:
        return "Low"
    elif prob >= 0.45:
        return "Medium"
    else:
        return "High"
