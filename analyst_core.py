import pandas as pd
import numpy as np
import io
from pathlib import Path

def _looks_like_csv(raw_bytes: bytes) -> bool:
    try:
        sample = raw_bytes[:1024].decode(errors="ignore")
    except Exception:
        return False
    return "," in sample and "\n" in sample

def load_data(file_or_path) -> pd.DataFrame:
    """
    Accepts path string/Path, or file-like object.
    Returns pandas DataFrame.
    """
    if isinstance(file_or_path, (str, Path)):
        p = Path(file_or_path)
        s = p.suffix.lower()
        if s == ".csv":
            return pd.read_csv(p)
        if s in {".xls", ".xlsx"}:
            return pd.read_excel(p)
        if s == ".json":
            return pd.read_json(p)
        return pd.read_csv(p)
    
    # file-like
    name = getattr(file_or_path, "name", None)
    suffix = Path(name).suffix.lower() if name else None
    raw = file_or_path.read()
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    bio = io.BytesIO(raw)
    
    if suffix == ".csv" or (suffix is None and _looks_like_csv(raw)):
        bio.seek(0); return pd.read_csv(bio)
    if suffix in {".xls", ".xlsx"}:
        bio.seek(0); return pd.read_excel(bio)
    if suffix == ".json":
        bio.seek(0); return pd.read_json(bio)
        
    # fallback
    bio.seek(0)
    try:
        return pd.read_csv(bio)
    except Exception:
        bio.seek(0); return pd.read_json(bio)

def _detect_column_types(df: pd.DataFrame):
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    datetime = []
    # try to infer datetime columns
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.datetime64):
            datetime.append(c)
        else:
            # try to parse small sample as date
            try:
                sample = df[c].dropna().astype(str).iloc[:20]
                parsed = pd.to_datetime(sample, errors="coerce")
                if parsed.notna().sum() >= max(1, min(5, len(sample)//2)):
                    datetime.append(c)
            except Exception:
                pass
    
    # categoricals: low cardinality non-numeric
    categorical = [c for c in df.columns if c not in numeric + datetime and df[c].nunique(dropna=True) <= 50]
    
    return {"numeric": numeric, "datetime": datetime, "categorical": categorical}

def generate_data_report(df: pd.DataFrame) -> str:
    types = _detect_column_types(df)
    
    report = []
    report.append(f"## Dataset Overview")
    report.append(f"- **Rows**: {len(df)}")
    report.append(f"- **Columns**: {len(df.columns)}")
    report.append(f"- **Column Names**: {', '.join(df.columns)}")
    
    report.append(f"\n## Column Types Detected")
    report.append(f"- **Numeric**: {', '.join(types['numeric']) if types['numeric'] else 'None'}")
    report.append(f"- **Datetime**: {', '.join(types['datetime']) if types['datetime'] else 'None'}")
    report.append(f"- **Categorical**: {', '.join(types['categorical']) if types['categorical'] else 'None'}")
    
    report.append(f"\n## Missing Values")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        for col, count in missing.items():
            report.append(f"- {col}: {count} missing")
    else:
        report.append("- No missing values found.")
        
    return "\n".join(report)

def parse_suffix(val):
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    val = str(val).upper().replace(",", "")
    if "K" in val:
        return float(val.replace("K", "")) * 1000
    if "M" in val:
        return float(val.replace("M", "")) * 1000000
    try:
        return float(val)
    except:
        return np.nan

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans specific columns known to have K/M suffixes.
    """
    cols_to_clean = ['Username_Followers', 'Collaborator_Followers']
    for col in cols_to_clean:
        if col in df.columns:
            df[col] = df[col].apply(parse_suffix)
    return df

def train_viral_predictor(df: pd.DataFrame):
    """
    Trains a simple Linear Regression model: Views ~ Followers.
    Returns: model, r2_score, X_test, y_test, y_pred
    """
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    
    # Prepare data
    data = df[['Username_Followers', 'views']].dropna()
    X = data[['Username_Followers']]
    y = data['views']
    
    if len(data) < 10:
        return None, 0, None, None, None
        
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)
    
    return model, score, X_test, y_test, y_pred
