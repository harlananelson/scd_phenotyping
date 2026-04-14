"""
Lab-Based Hemoglobinopathy Phenotyping
======================================

Full classification based on hemoglobin electrophoresis / HPLC results.
Implements the algorithm from kwuichet/SCD_Phenotyping with:
- 9 inter-subunit ratios (AoverS, CoverS, DoverS, EoverS, VoverS, AoverC, AoverD, AoverE, AoverV)
- Age-stratified classification (< 2 years vs >= 2 years)
- Transfusion-aware interpretation (inferred from longitudinal HgbA/HgbS variation)
- SPFH detection (S-pattern with persistent fetal hemoglobin)
- Fractionation completeness checks (90-105% sum)
- Person-level priority cascade (NOT majority vote)

Reference: kwuichet/SCD_Phenotyping hemoglobinopathy_phenotyping_notebook.ipynb
"""

import re
import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Required hemoglobin subunits
REQUIRED_SUBS = ['HgbA', 'HgbS', 'HgbF', 'HgbC']

# Optional hemoglobin subunits
OPTIONAL_SUBS = ['HgbA2', 'HgbD', 'HgbE', 'HgbO']

ALL_SUBS = REQUIRED_SUBS + OPTIONAL_SUBS

# ---------------------------------------------------------------------------
# Clinical Thresholds (documented with rationale)
# Reference: kwuichet/SCD_PHenotyping algorithm
# ---------------------------------------------------------------------------

# Fractionation completeness bounds (percentage)
# A complete Hb electrophoresis sums to ~100%. Range accounts for lab variability.
FRAC_MIN = 90
FRAC_MAX = 105

# Transfusion inference threshold (percentage point range in HgbA or HgbS)
# If max-min range >= 10pp, patient likely received transfusions
TFX_RANGE_THRESHOLD = 10

# Post-transfusion threshold (days since actual transfusion)
# Within 180 days, transfused RBCs contribute HbA, altering fractionation
TFX_DAYS_THRESHOLD = 180

# Post-hydroxyurea threshold (days since HU prescription)
# HU raises HbF; within 90 days, HbF levels may reflect treatment not genotype
HU_DAYS_THRESHOLD = 90

# Lab error threshold: HgbS values <= this considered clinically absent
# Accounts for HPLC/electrophoresis measurement noise
LAB_ERROR_THRESHOLD = 5.0

# --- Compound hemoglobinopathy ratios (Step 1) ---
# SC disease: HbC and HbS are co-dominant, expected CoverS ~1.0
# Range 0.6-1.4 accounts for expression variability
SC_RATIO_LOWER = 0.6
SC_RATIO_UPPER = 1.4

# SD disease: same co-dominant logic as SC
SD_RATIO_LOWER = 0.6
SD_RATIO_UPPER = 1.4

# SE disease: HbE typically 20-30% of total, HbS 50-60%
# Lower ratio range because HbE expression is inherently lower
SE_RATIO_LOWER = 0.3
SE_RATIO_UPPER = 0.7

# SVar (other variants): same range as SC/SD
SVAR_RATIO_LOWER = 0.6
SVAR_RATIO_UPPER = 1.4

# --- Non-SCD hemoglobinopathy ratios (Step 2) ---
# Trait: A/variant ratio 0.9-3.25 (heterozygous carrier)
# Disease: A/variant ratio < 0.5 (homozygous or compound het)
# Indeterminate: 0.5-0.9 (borderline)
NON_SCD_TRAIT_RANGE = (0.9, 3.25)
NON_SCD_DISEASE_UPPER = 0.5
NON_SCD_INDET_RANGE = (0.5, 0.9)

# Beta-thalassemia: HbA2 > 3.5% is standard clinical cutoff
# Using 4% for specificity in a phenotyping (not diagnostic) context
BETA_THAL_HBA2_THRESHOLD = 4.0

# --- Sickle trait identification ---
# In sickle trait, HbA > HbS (typically 55-65% A, 30-40% S)
# A/S ratio between 1.0 and 2.0
TRAIT_AS_RATIO_LOWER = 1.0
TRAIT_AS_RATIO_UPPER = 2.0

# --- Incomplete fractionation heuristics (Step 4b) ---
# When fractionation is incomplete, use absolute HgbS level
HBS_DEFINITIVE_SCA = 80    # Very high S → definitive SCA
HBS_LIKELY_SCA = 70         # High S → likely SCA
HBS_POSSIBLE_SCA = 50       # Moderate S → needs context (A level, tfx status)
HBS_INDETERMINATE = 60      # SCD_Indeterminate threshold
HBS_S_INDETERMINATE = 50    # S_Indeterminate threshold

# --- SPFH detection ---
# S-pattern with persistent fetal hemoglobin
# Requires: HbF >= 20%, HbS >= 40%, total Hb > 12 g/dL, not on HU
SPFH_HBF_MIN = 20
SPFH_HBS_MIN = 40
SPFH_TOTAL_HB_MIN = 12

# Ratio definitions: (numerator_col, denominator_col)
RATIO_DEFINITIONS: Dict[str, Tuple[str, str]] = {
    'AoverC': ('HgbA', 'HgbC'),
    'AoverD': ('HgbA', 'HgbD'),
    'AoverE': ('HgbA', 'HgbE'),
    'AoverV': ('HgbA', 'HgbO'),
    'AoverS': ('HgbA', 'HgbS'),
    'CoverS': ('HgbC', 'HgbS'),
    'DoverS': ('HgbD', 'HgbS'),
    'EoverS': ('HgbE', 'HgbS'),
    'VoverS': ('HgbO', 'HgbS'),
}

# Compound hemoglobinopathy ratio thresholds
COMPOUND_THRESHOLDS = {
    'CoverS': (0.6, 1.4),   # SCD_SC
    'DoverS': (0.6, 1.4),   # SCD_SD
    'EoverS': (0.3, 0.7),   # SCD_SE
    'VoverS': (0.6, 1.4),   # SCD_SVar
}

# Non-SCD hemoglobinopathy A-over-X thresholds
NON_SCD_TRAIT_RANGE = (0.9, 3.25)     # Trait
NON_SCD_DISEASE_UPPER = 0.5           # Disease (< 0.5)
NON_SCD_INDET_RANGE = (0.5, 0.9)     # Indeterminate


# ---------------------------------------------------------------------------
# Data Preparation Utilities
# ---------------------------------------------------------------------------

def extract_numeric(x) -> Optional[float]:
    """
    Extract numeric value from potentially non-numeric strings.

    Handles formats like ">90", "<1", "45.2%", "~30", etc.

    Parameters
    ----------
    x : any
        Value to extract number from.

    Returns
    -------
    float or None
        Extracted numeric value, or None if no number found.
    """
    if pd.isna(x):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    matches = re.findall(r'\d+\.?\d*', str(x))
    if matches:
        return float(matches[0])
    return None


def pivot_lab_data(
    df: pd.DataFrame,
    id_col: str = 'personid',
    date_col: str = 'date',
    age_col: str = 'age',
    hgb_type_col: str = 'hgbType',
    value_col: str = 'value'
) -> pd.DataFrame:
    """
    Convert key-value (long) lab data to wide format with one column per Hgb subunit.

    Input format: personid | date | age | hgbType | value
    Output format: personid | date | age | HgbA | HgbS | HgbF | HgbC | ...

    Handles duplicates by keeping the value closest to expected range.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format lab data.
    id_col, date_col, age_col, hgb_type_col, value_col : str
        Column name mappings.

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame with hemoglobin subunit columns.
    """
    df = df.copy()

    # Extract numeric values
    df[value_col] = df[value_col].apply(extract_numeric)
    df = df.dropna(subset=[value_col])

    # Pivot: keep first occurrence per person-date-hgbType
    deduped = (
        df
        .sort_values([id_col, date_col, hgb_type_col, value_col])
        .drop_duplicates(subset=[id_col, date_col, hgb_type_col], keep='first')
    )

    index_cols = [id_col, date_col]
    if age_col is not None and age_col in deduped.columns:
        index_cols.append(age_col)

    wide = deduped.pivot_table(
        index=index_cols,
        columns=hgb_type_col,
        values=value_col,
        aggfunc='first'
    ).reset_index()

    # Flatten MultiIndex columns if needed
    wide.columns = [c if isinstance(c, str) else c for c in wide.columns]

    return wide


def get_best_hgb_row(
    df: pd.DataFrame,
    id_col: str = 'personid',
    date_col: str = 'date'
) -> pd.DataFrame:
    """
    When multiple fractionation results exist for the same person-date,
    keep the row closest to 100% total.

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format lab data with HgbSum and CompleteFrac columns.

    Returns
    -------
    pd.DataFrame
        Deduplicated DataFrame.
    """
    if 'HgbSum' not in df.columns:
        return df

    df = df.copy()
    df['_dist_from_100'] = (df['HgbSum'] - 100).abs()

    # Prefer CompleteFrac == 'Y', then closest to 100
    df['_complete_rank'] = df['CompleteFrac'].map({'Y': 0, 'N': 1}).fillna(1)

    result = (
        df
        .sort_values([id_col, date_col, '_complete_rank', '_dist_from_100'])
        .drop_duplicates(subset=[id_col, date_col], keep='first')
        .drop(columns=['_dist_from_100', '_complete_rank'])
    )

    return result


# ---------------------------------------------------------------------------
# Ratio and Completeness Calculations
# ---------------------------------------------------------------------------

def calculate_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate hemoglobin fraction percentages and inter-subunit ratios.

    Computes:
    - S_percent, A_percent, F_percent, C_percent (backward-compatible)
    - HgbSum (total of all present subunits)
    - CompleteFrac ('Y' if HgbSum in [90, 105], 'N' otherwise)
    - 9 inter-subunit ratios: AoverS, CoverS, DoverS, EoverS, VoverS,
      AoverC, AoverD, AoverE, AoverV

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format lab data with hemoglobin subunit columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with added ratio and completeness columns.
    """
    df = df.copy()

    # Identify which subunit columns are present
    present_subs = [c for c in ALL_SUBS if c in df.columns]
    if not present_subs:
        raise ValueError(f"Expected at least one of {ALL_SUBS} columns")

    # Total hemoglobin sum
    df['HgbSum'] = df[present_subs].sum(axis=1, min_count=1)

    # Completeness flag
    df['CompleteFrac'] = np.where(
        df['HgbSum'].between(FRAC_MIN, FRAC_MAX),
        'Y', 'N'
    )

    # Simple percentages (backward-compatible)
    mask = df['HgbSum'] > 0
    for sub in ['HgbS', 'HgbA', 'HgbF', 'HgbC']:
        pct_col = sub.replace('Hgb', '') + '_percent'
        if sub in df.columns:
            df[pct_col] = np.where(mask, (df[sub] / df['HgbSum']) * 100, np.nan)

    # Inter-subunit ratios
    for ratio_name, (num_col, den_col) in RATIO_DEFINITIONS.items():
        if num_col in df.columns and den_col in df.columns:
            num_valid = df[num_col].fillna(0) >= 0
            den_valid = df[den_col].fillna(0) > 0
            both_valid = num_valid & den_valid

            df[ratio_name] = np.nan
            df.loc[both_valid, ratio_name] = (
                df.loc[both_valid, num_col] / df.loc[both_valid, den_col]
            )

            # If numerator is null/0 but denominator > 0 and CompleteFrac == Y, ratio = 0
            complete_zero_num = (
                (df[num_col].fillna(0) == 0)
                & den_valid
                & (df['CompleteFrac'] == 'Y')
            )
            df.loc[complete_zero_num, ratio_name] = 0.0

    return df


# ---------------------------------------------------------------------------
# Transfusion Inference
# ---------------------------------------------------------------------------

def infer_transfusion_status(
    df: pd.DataFrame,
    id_col: str = 'personid'
) -> pd.DataFrame:
    """
    Infer transfusion status from longitudinal HgbA/HgbS variation.

    If a person's HgbA or HgbS varies by >= 10 percentage points across labs,
    transfusion is suspected. Rows where HgbA is elevated above the person's
    minimum are tagged as likely post-transfusion.

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format lab data with HgbA, HgbS columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with InferTfxPerson and InferTfxRow columns.
    """
    df = df.copy()

    if 'HgbA' not in df.columns or 'HgbS' not in df.columns:
        df['InferTfxPerson'] = 'No'
        df['InferTfxRow'] = 'Unk'
        return df

    # Person-level stats
    person_stats = df.groupby(id_col).agg(
        HgbA_min=('HgbA', 'min'),
        HgbA_max=('HgbA', 'max'),
        HgbS_min=('HgbS', 'min'),
        HgbS_max=('HgbS', 'max'),
    ).reset_index()

    person_stats['HgbA_range'] = person_stats['HgbA_max'] - person_stats['HgbA_min']
    person_stats['HgbS_range'] = person_stats['HgbS_max'] - person_stats['HgbS_min']

    person_stats['InferTfxPerson'] = np.where(
        (person_stats['HgbA_range'] >= TFX_RANGE_THRESHOLD)
        | (person_stats['HgbS_range'] >= TFX_RANGE_THRESHOLD),
        'Yes', 'No'
    )

    # Merge back
    df = df.merge(
        person_stats[[id_col, 'InferTfxPerson', 'HgbA_min', 'HgbS_min', 'HgbS_max']],
        on=id_col,
        how='left'
    )

    # Row-level inference
    conditions = [
        (df['InferTfxPerson'] == 'Yes') & (df['HgbA'] >= df['HgbA_min'] + TFX_RANGE_THRESHOLD),
        (df['InferTfxPerson'] == 'Yes') & (df['HgbA'] < df['HgbA_min'] + TFX_RANGE_THRESHOLD),
    ]
    choices = ['Post', 'Pre']
    df['InferTfxRow'] = np.select(conditions, choices, default='Unk')

    # Also check S-based inference
    s_post = (
        (df['InferTfxPerson'] == 'Yes')
        & (df['HgbS'] < df['HgbS_min'] + TFX_RANGE_THRESHOLD)
        & (df['InferTfxRow'] == 'Unk')
    )
    df.loc[s_post, 'InferTfxRow'] = 'PostMaybe'

    df = df.drop(columns=['HgbA_min', 'HgbS_min', 'HgbS_max'], errors='ignore')

    return df


def add_actual_transfusion(
    df: pd.DataFrame,
    tfx_df: Optional[pd.DataFrame] = None,
    id_col: str = 'personid',
    date_col: str = 'date',
    tfx_date_col: str = 'transfusion_date',
    threshold_days: int = TFX_DAYS_THRESHOLD
) -> pd.DataFrame:
    """
    Add actual transfusion proximity flags from transfusion records.

    Parameters
    ----------
    df : pd.DataFrame
        Lab data with date column.
    tfx_df : pd.DataFrame, optional
        Transfusion records with id and date columns. If None, all rows get PostTfx='N'.
    threshold_days : int
        Days after transfusion to consider "post-transfusion".

    Returns
    -------
    pd.DataFrame
        DataFrame with DaysTfx and PostTfx columns.
    """
    df = df.copy()

    if tfx_df is None or tfx_df.empty:
        # No actual transfusion file — use inferred status from HgbA/HgbS ranges
        # Reference logic: InferTfxRow 'Post' or 'PostMaybe' → PostTfx='Y'
        df['DaysTfx'] = 99999
        if 'InferTfxRow' in df.columns:
            df['PostTfx'] = np.where(
                df['InferTfxRow'].isin(['Post', 'PostMaybe']), 'Y', 'N'
            )
        else:
            df['PostTfx'] = 'N'
        return df

    # Merge lab dates with transfusion dates (on or before lab date)
    merged = df[[id_col, date_col]].drop_duplicates().merge(
        tfx_df[[id_col, tfx_date_col]],
        on=id_col,
        how='left'
    )

    merged[date_col] = pd.to_datetime(merged[date_col])
    merged[tfx_date_col] = pd.to_datetime(merged[tfx_date_col])

    # Keep only transfusions on or before lab date
    merged = merged[merged[tfx_date_col] <= merged[date_col]]

    # Get most recent transfusion per person-date
    latest_tfx = (
        merged
        .groupby([id_col, date_col])[tfx_date_col]
        .max()
        .reset_index()
    )
    latest_tfx['DaysTfx'] = (
        latest_tfx[date_col] - latest_tfx[tfx_date_col]
    ).dt.days

    df = df.merge(
        latest_tfx[[id_col, date_col, 'DaysTfx']],
        on=[id_col, date_col],
        how='left'
    )
    df['DaysTfx'] = df['DaysTfx'].fillna(99999).astype(int)
    df['PostTfx'] = np.where(df['DaysTfx'] <= threshold_days, 'Y', 'N')

    return df


def add_hu_proximity(
    df: pd.DataFrame,
    hu_df: Optional[pd.DataFrame] = None,
    id_col: str = 'personid',
    date_col: str = 'date',
    hu_date_col: str = 'hu_date',
    threshold_days: int = HU_DAYS_THRESHOLD
) -> pd.DataFrame:
    """
    Add hydroxyurea prescription proximity flags.

    Parameters
    ----------
    df : pd.DataFrame
        Lab data.
    hu_df : pd.DataFrame, optional
        HU prescription dates. If None, all rows get PostHU='N'.

    Returns
    -------
    pd.DataFrame
        DataFrame with DaysHU and PostHU columns.
    """
    df = df.copy()

    if hu_df is None or hu_df.empty:
        df['DaysHU'] = 99999
        df['PostHU'] = 'N'
        return df

    merged = df[[id_col, date_col]].drop_duplicates().merge(
        hu_df[[id_col, hu_date_col]],
        on=id_col,
        how='left'
    )

    merged[date_col] = pd.to_datetime(merged[date_col])
    merged[hu_date_col] = pd.to_datetime(merged[hu_date_col])

    merged = merged[merged[hu_date_col] <= merged[date_col]]

    latest_hu = (
        merged
        .groupby([id_col, date_col])[hu_date_col]
        .max()
        .reset_index()
    )
    latest_hu['DaysHU'] = (
        latest_hu[date_col] - latest_hu[hu_date_col]
    ).dt.days

    df = df.merge(
        latest_hu[[id_col, date_col, 'DaysHU']],
        on=[id_col, date_col],
        how='left'
    )
    df['DaysHU'] = df['DaysHU'].fillna(99999).astype(int)
    df['PostHU'] = np.where(df['DaysHU'] <= threshold_days, 'Y', 'N')

    return df


# ---------------------------------------------------------------------------
# Row-Level Classification
# ---------------------------------------------------------------------------

def _get_val(row, col, default=0.0):
    """Safely get a float value from a row, returning default if missing/NaN."""
    val = row.get(col)
    if pd.isna(val):
        return default
    return float(val)


def _get_ratio(row, ratio_name):
    """Safely get a ratio value, returning None if missing."""
    val = row.get(ratio_name)
    if pd.isna(val):
        return None
    return float(val)


def _in_range(val, low, high):
    """Check if val is in [low, high] inclusive. Returns False if val is None."""
    if val is None:
        return False
    return low <= val <= high


def classify_lab_row(
    row: pd.Series,
    dataset_scd: bool = False,
    run_spfh: bool = False,
    avg_hgb_total: Optional[float] = None
) -> str:
    """
    Classify a single lab result row based on hemoglobin fractions and ratios.

    Implements the full decision tree from the upstream phenotyping notebook
    with age-stratified logic, transfusion awareness, and SPFH detection.

    Parameters
    ----------
    row : pd.Series
        Single row with hemoglobin values, ratios, and context columns.
        Expected columns: HgbA, HgbS, HgbF, HgbC, HgbD, HgbE, HgbO,
        AoverS, CoverS, DoverS, EoverS, VoverS, AoverC, AoverD, AoverE, AoverV,
        HgbSum, CompleteFrac, PostTfx, PostHU, age, HgbA2.
    dataset_scd : bool
        If True, we are classifying within a known-SCD dataset (skip non-SCD checks).
    run_spfh : bool
        If True, check for S-pattern with persistent fetal hemoglobin.
    avg_hgb_total : float, optional
        Running average total hemoglobin for SPFH check. If None, uses 0.

    Returns
    -------
    str
        Classification label.
    """
    # Extract values
    hgb_s = _get_val(row, 'HgbS', 0)
    hgb_a = _get_val(row, 'HgbA', 0)
    hgb_f = _get_val(row, 'HgbF', 0)
    hgb_c = _get_val(row, 'HgbC', 0)
    hgb_d = _get_val(row, 'HgbD', 0)
    hgb_e = _get_val(row, 'HgbE', 0)
    hgb_o = _get_val(row, 'HgbO', 0)
    hgb_a2 = _get_val(row, 'HgbA2', 0)
    hgb_sum = _get_val(row, 'HgbSum', 0)

    complete = row.get('CompleteFrac', 'N')
    post_tfx = row.get('PostTfx', 'N')
    post_hu = row.get('PostHU', 'N')
    age = _get_val(row, 'age', None)

    avg_total = avg_hgb_total if avg_hgb_total is not None else 0

    # Ratios
    a_over_s = _get_ratio(row, 'AoverS')
    c_over_s = _get_ratio(row, 'CoverS')
    d_over_s = _get_ratio(row, 'DoverS')
    e_over_s = _get_ratio(row, 'EoverS')
    v_over_s = _get_ratio(row, 'VoverS')
    a_over_c = _get_ratio(row, 'AoverC')
    a_over_d = _get_ratio(row, 'AoverD')
    a_over_e = _get_ratio(row, 'AoverE')
    a_over_v = _get_ratio(row, 'AoverV')

    # ---- STEP 1: Compound hemoglobinopathies (any age) ----
    if _in_range(c_over_s, SC_RATIO_LOWER, SC_RATIO_UPPER):
        return 'SCD_SC'
    if _in_range(d_over_s, SD_RATIO_LOWER, SD_RATIO_UPPER):
        return 'SCD_SD'
    if _in_range(e_over_s, SE_RATIO_LOWER, SE_RATIO_UPPER):
        return 'SCD_SE'
    if _in_range(v_over_s, SVAR_RATIO_LOWER, SVAR_RATIO_UPPER):
        return 'SCD_SVar'

    # ---- STEP 2: Non-SCD hemoglobinopathies (only if not SCD-only dataset) ----
    if not dataset_scd and complete == 'Y' and hgb_s <= LAB_ERROR_THRESHOLD:
        if post_tfx == 'N':
            # HemC
            if a_over_c is not None:
                if _in_range(a_over_c, *NON_SCD_TRAIT_RANGE):
                    return 'HemC_Trait'
                if a_over_c < NON_SCD_DISEASE_UPPER:
                    return 'HemC_Disease'
                if _in_range(a_over_c, *NON_SCD_INDET_RANGE):
                    return 'HemC_Indeterminate'
            # HemD
            if a_over_d is not None:
                if _in_range(a_over_d, *NON_SCD_TRAIT_RANGE):
                    return 'HemD_Trait'
                if a_over_d < NON_SCD_DISEASE_UPPER:
                    return 'HemD_Disease'
                if _in_range(a_over_d, *NON_SCD_INDET_RANGE):
                    return 'HemD_Indeterminate'
            # HemE
            if a_over_e is not None:
                if _in_range(a_over_e, *NON_SCD_TRAIT_RANGE):
                    return 'HemE_Trait'
                if a_over_e < NON_SCD_DISEASE_UPPER:
                    return 'HemE_Disease'
                if _in_range(a_over_e, *NON_SCD_INDET_RANGE):
                    return 'HemE_Indeterminate'
            # Beta Thalassemia
            if hgb_a2 > BETA_THAL_HBA2_THRESHOLD:
                return 'BetaThalassemia'
            # Not SCD
            if age is not None and age >= 2 and hgb_s == 0 and hgb_o == 0:
                return 'Not_SCD'
        else:
            # Post-transfusion with no HgbS — limited classification
            if hgb_c > 0:
                return 'HemC_Indeterminate'
            if hgb_d > 0:
                return 'HemD_Indeterminate'
            if hgb_e > 0:
                return 'HemE_Indeterminate'

    # ---- STEP 3: Age < 2 (pediatric branch) ----
    if age is not None and age < 2:
        if hgb_s >= LAB_ERROR_THRESHOLD:
            if complete == 'Y':
                if hgb_a <= 1:
                    return 'SCD_SCA_Likely'
                if a_over_s is not None and a_over_s < SC_RATIO_LOWER:
                    if post_tfx == 'Y':
                        return 'SCD_SCA_Likely'
                    return 'SCD_Indeterminate'
                if a_over_s is not None and _in_range(a_over_s, SC_RATIO_LOWER, TRAIT_AS_RATIO_UPPER):
                    if post_tfx == 'N':
                        return 'S_Trait'
                    return 'S_Indeterminate'
            return 'No_Phenotype'
        return 'No_Phenotype'

    # ---- STEP 4: Age >= 2 (or age unknown — treated as adult) ----
    if complete == 'Y':
        # SPFH check
        if run_spfh and hgb_f >= SPFH_HBF_MIN and hgb_s >= SPFH_HBS_MIN and avg_total > SPFH_TOTAL_HB_MIN and post_hu == 'N':
            return 'SCD_SPFH'

        # Variant hemoglobin (no HgbS, but HgbO present)
        if hgb_s == 0 and (
            (a_over_v is not None and a_over_v <= NON_SCD_TRAIT_RANGE[1])
            or (hgb_o > 0 and hgb_a == 0)
        ):
            return 'Hem_Variant'

        # Definitive SCD SCA: high S, no/very low A, no O
        if hgb_s >= LAB_ERROR_THRESHOLD and hgb_a <= LAB_ERROR_THRESHOLD and hgb_o == 0:
            return 'SCD_SCA'

        # High S, moderate A — needs transfusion disambiguation
        if hgb_s >= 40 and hgb_a < 40:
            if post_tfx == 'Y':
                return 'SCD_SCA_Likely'
            return 'SCD_Sbetap_Likely'

        # Sickle trait pattern: A > S
        if hgb_s >= LAB_ERROR_THRESHOLD and a_over_s is not None and _in_range(a_over_s, TRAIT_AS_RATIO_LOWER, TRAIT_AS_RATIO_UPPER) and post_tfx == 'N':
            return 'S_Trait'

        # S present but inconclusive
        if hgb_s >= LAB_ERROR_THRESHOLD:
            return 'S_Indeterminate'

    # ---- Incomplete fractionation fallbacks ----
    if hgb_sum < FRAC_MIN and hgb_sum > 0:
        if hgb_s >= HBS_DEFINITIVE_SCA:
            return 'SCD_SCA'
        if hgb_s >= HBS_LIKELY_SCA:
            return 'SCD_SCA_Likely'
        if hgb_s >= HBS_POSSIBLE_SCA and hgb_a <= LAB_ERROR_THRESHOLD:
            return 'SCD_SCA_Likely'
        if hgb_s >= HBS_POSSIBLE_SCA and post_tfx == 'Y':
            return 'SCD_SCA_Likely'
        if hgb_s >= HBS_INDETERMINATE:
            return 'SCD_Indeterminate'
        if hgb_s >= HBS_S_INDETERMINATE:
            return 'S_Indeterminate'

    if hgb_sum > FRAC_MAX:
        return 'No_Phenotype'

    # ---- SCD-only dataset fallback ----
    if dataset_scd and hgb_s >= 5:
        return 'S_Present'

    return 'No_Phenotype'


# ---------------------------------------------------------------------------
# Person-Level Aggregation (Priority Cascade)
# ---------------------------------------------------------------------------

def aggregate_lab_phenotypes(
    lab_df: pd.DataFrame,
    id_col: str = 'personid',
    pheno_col: str = 'LabPhenotype',
    dataset_scd: bool = False
) -> pd.DataFrame:
    """
    Aggregate row-level phenotypes to patient level using priority cascade.

    Uses the upstream algorithm's strict priority hierarchy (NOT majority vote).
    Compound hemoglobinopathies take precedence; transfusion percentage
    disambiguates between SCA-Likely and Sbetap-Likely.

    Parameters
    ----------
    lab_df : pd.DataFrame
        DataFrame with per-test phenotype classifications.
    id_col : str
        Column name for person identifier.
    pheno_col : str
        Column name containing phenotype labels.
    dataset_scd : bool
        If True, use SCD-only dataset logic for trait/indeterminate resolution.

    Returns
    -------
    pd.DataFrame
        One row per person with columns: id_col, LabPhenotype, phenotype counts,
        HgbSMax, HgbAMax, TfxPercent.
    """
    # Build phenotype count matrix
    pheno_matrix = (
        lab_df
        .groupby(id_col)[pheno_col]
        .value_counts()
        .unstack(fill_value=0)
        .reset_index()
    )

    # Add HgbS/HgbA max and transfusion percentage
    if 'HgbS' in lab_df.columns:
        hgb_s_max = lab_df.groupby(id_col)['HgbS'].max().reset_index()
        hgb_s_max.columns = [id_col, 'HgbSMax']
        pheno_matrix = pheno_matrix.merge(hgb_s_max, on=id_col, how='left')
    else:
        pheno_matrix['HgbSMax'] = 0

    if 'HgbA' in lab_df.columns:
        hgb_a_max = lab_df.groupby(id_col)['HgbA'].max().reset_index()
        hgb_a_max.columns = [id_col, 'HgbAMax']
        pheno_matrix = pheno_matrix.merge(hgb_a_max, on=id_col, how='left')
    else:
        pheno_matrix['HgbAMax'] = 0

    # Transfusion percent: fraction of S-positive rows that are post-transfusion
    if 'PostTfx' in lab_df.columns and 'HgbS' in lab_df.columns:
        s_positive = lab_df[lab_df['HgbS'].fillna(0) > 0]
        if len(s_positive) > 0:
            tfx_pct = (
                s_positive
                .groupby(id_col)
                .apply(
                    lambda g: (g['PostTfx'] == 'Y').sum() / len(g) * 100
                )
                .reset_index()
            )
            tfx_pct.columns = [id_col, 'TfxPercent']
            pheno_matrix = pheno_matrix.merge(tfx_pct, on=id_col, how='left')
        else:
            pheno_matrix['TfxPercent'] = 0
    else:
        pheno_matrix['TfxPercent'] = 0

    pheno_matrix['TfxPercent'] = pheno_matrix['TfxPercent'].fillna(0)

    # Apply priority cascade
    pheno_matrix['LabPhenotype'] = pheno_matrix.apply(
        lambda r: _assign_person_phenotype(r, dataset_scd), axis=1
    )

    # Return clean result
    return pheno_matrix[[id_col, 'LabPhenotype', 'HgbSMax', 'HgbAMax', 'TfxPercent']]


def _count(row, label):
    """Get count of a phenotype label from the matrix row, defaulting to 0."""
    return row.get(label, 0)


def _assign_person_phenotype(row: pd.Series, dataset_scd: bool = False) -> str:
    """
    Assign person-level phenotype using strict priority cascade.

    Parameters
    ----------
    row : pd.Series
        Row from phenotype count matrix.
    dataset_scd : bool
        If True, use SCD-only dataset mode.

    Returns
    -------
    str
        Person-level phenotype label.
    """
    tfx_pct = row.get('TfxPercent', 0)

    # ---- TIER 1: Definitive compound hemoglobinopathies ----
    if _count(row, 'SCD_SC') > 0:
        return 'SCD_SC'
    if _count(row, 'SCD_SD') > 0:
        return 'SCD_SD'
    if _count(row, 'SCD_SE') > 0:
        return 'SCD_SE'
    if _count(row, 'SCD_SPFH') > 0:
        return 'SCD_SPFH_Likely'
    if _count(row, 'SCD_SCA') > 0:
        return 'SCD_SCA'
    if _count(row, 'SCD_SVar') > 0:
        return 'SCD_SVar'

    # ---- TIER 2: Non-SCD hemoglobinopathies ----
    if not dataset_scd:
        for hem_type in ['HemC', 'HemD', 'HemE']:
            if _count(row, f'{hem_type}_Disease') > 0:
                return f'{hem_type}_Disease'
            if _count(row, f'{hem_type}_Trait') > 0:
                return f'{hem_type}_Trait'
            if _count(row, f'{hem_type}_Indeterminate') > 0:
                return f'{hem_type}_Indeterminate'
        if _count(row, 'BetaThalassemia') > 0:
            return 'BetaThalassemia'

    # ---- TIER 3: SCD likely/indeterminate (transfusion-disambiguated) ----
    sbetap = _count(row, 'SCD_Sbetap_Likely')
    sca_likely = _count(row, 'SCD_SCA_Likely')
    s_indet = _count(row, 'S_Indeterminate')
    scd_indet = _count(row, 'SCD_Indeterminate')

    if sbetap > 0 and sca_likely == 0 and s_indet == 0:
        return 'SCD_Sbetap_Likely'

    if sbetap > 0 and (sca_likely > 0 or s_indet > 0):
        if tfx_pct >= 50:
            return 'SCD_SCA_Likely'
        return 'SCD_Sbetap_Likely'

    if sca_likely > 0:
        return 'SCD_SCA_Likely'

    if (s_indet > 0 or scd_indet > 0) and tfx_pct >= 50:
        return 'SCD_SCA_Likely'

    if scd_indet > 0:
        return 'SCD_Indeterminate'

    if _count(row, 'Hem_Variant') > 0:
        return 'Hem_Variant'

    # ---- TIER 4: Trait/Non-SCD ----
    if not dataset_scd:
        s_trait = _count(row, 'S_Trait')
        if s_trait > 0 and tfx_pct == 0:
            return 'S_Trait'
        if s_trait > 0 and tfx_pct > 0:
            return 'No_Phenotype'
        if _count(row, 'Not_SCD') > 0:
            return 'Not_SCD'
        if s_indet > 0:
            return 'S_Indeterminate'
    else:
        # SCD-only dataset mode
        if _count(row, 'S_Present') > 0 and tfx_pct >= 50:
            return 'SCD_SCA_Likely'
        if s_indet > 0:
            return 'SCD_Indeterminate'

    return 'No_Phenotype'


# ---------------------------------------------------------------------------
# Top-Level Orchestrator
# ---------------------------------------------------------------------------

def run_lab_phenotyping(
    lab_df: pd.DataFrame,
    id_col: str = 'personid',
    date_col: str = 'date',
    age_col: str = 'age',
    hgb_type_col: str = 'hgbType',
    value_col: str = 'value',
    input_format: str = 'long',
    tfx_df: Optional[pd.DataFrame] = None,
    hu_df: Optional[pd.DataFrame] = None,
    dataset_scd: bool = False,
    run_spfh: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the full lab-based phenotyping pipeline.

    Parameters
    ----------
    lab_df : pd.DataFrame
        Lab data. If input_format='long', expects columns: id, date, age,
        hgbType, value. If input_format='wide', expects columns with Hgb subunits.
    id_col, date_col, age_col, hgb_type_col, value_col : str
        Column name mappings for long format.
    input_format : str
        'long' (key-value) or 'wide' (one column per Hgb subunit).
    tfx_df : pd.DataFrame, optional
        Transfusion records for actual transfusion dating.
    hu_df : pd.DataFrame, optional
        Hydroxyurea prescription records for SPFH detection.
    dataset_scd : bool
        If True, treat as SCD-only dataset.
    run_spfh : bool
        If True, check for persistent fetal hemoglobin pattern.

    Returns
    -------
    tuple of (person_df, row_df)
        person_df: One row per person with LabPhenotype.
        row_df: All lab rows with LabPhenotype classification.
    """
    logger.info("Starting lab-based phenotyping pipeline")

    # Step 1: Pivot if needed
    if input_format == 'long':
        logger.info("Pivoting long-format lab data to wide format")
        wide_df = pivot_lab_data(lab_df, id_col, date_col, age_col, hgb_type_col, value_col)
    else:
        wide_df = lab_df.copy()

    logger.info(f"Wide-format data: {len(wide_df)} rows, {len(wide_df.columns)} columns")

    # Step 2: Calculate ratios and completeness
    wide_df = calculate_ratios(wide_df)

    # Step 3: Deduplicate same-day results
    wide_df = get_best_hgb_row(wide_df, id_col, date_col)
    logger.info(f"After deduplication: {len(wide_df)} rows")

    # Step 4: Infer transfusion status
    wide_df = infer_transfusion_status(wide_df, id_col)

    # Step 5: Add actual transfusion proximity
    wide_df = add_actual_transfusion(wide_df, tfx_df, id_col, date_col)

    # Step 6: Add HU proximity (for SPFH)
    if run_spfh:
        wide_df = add_hu_proximity(wide_df, hu_df, id_col, date_col)
    else:
        wide_df['PostHU'] = 'N'

    # Step 7: Row-level classification
    logger.info("Classifying individual lab rows")
    wide_df['LabPhenotype'] = wide_df.apply(
        lambda r: classify_lab_row(r, dataset_scd=dataset_scd, run_spfh=run_spfh),
        axis=1
    )

    row_counts = wide_df['LabPhenotype'].value_counts()
    logger.info(f"Row-level phenotype distribution:\n{row_counts.to_string()}")

    # Step 8: Person-level aggregation via priority cascade
    logger.info("Aggregating to person-level phenotypes")
    person_df = aggregate_lab_phenotypes(
        wide_df, id_col=id_col, pheno_col='LabPhenotype', dataset_scd=dataset_scd
    )

    person_counts = person_df['LabPhenotype'].value_counts()
    logger.info(f"Person-level phenotype distribution:\n{person_counts.to_string()}")

    return person_df, wide_df
