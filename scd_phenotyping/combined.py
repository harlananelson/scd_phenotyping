"""
Combined ICD + Lab Phenotyping
==============================

Outer joins ICD-based and lab-based phenotype assignments by person.
Keeps BOTH columns (IcdPheno + PersonPhenotype) — no consensus merge.
Business rules for deriving scd_type (HbSS/HbSB0, HbSC, Control)
are applied downstream in R targets, not here.

Reference: 050-Hemoglobinopath_Phenotyping-Run.ipynb (Stage 3: Combine)

The reference logic:
    combined = (
        e.phenoMatrix.df.select(['personid', 'PersonPhenotype'])
        .join(e.phenoICD.df.select(['personid','IcdPheno']),
              on=['personid'], how='outer')
    )
    extractList = combined.filter(
        F.col('PersonPhenotype').isin(labList) | F.col('IcdPheno').isin(icdList)
    )
"""

import logging
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Lab phenotypes that indicate SCD (from reference 050 notebook)
SCD_LAB_PHENOTYPES = [
    'SCD_SCA',
    'SCD_SC',
    'SCD_SD',
    'SCD_SE',
    'SCD_SVar',
    'SCD_SCA_Likely',
    'SCD_Sbetap_Likely',
    'SCD_SPFH_Likely',
    'SCD_Indeterminate',
]

# ICD phenotypes that indicate SCD (from reference 050 notebook)
SCD_ICD_PHENOTYPES = ['SCD']


def run_combined_phenotyping(
    icd_df: pd.DataFrame,
    lab_df: pd.DataFrame,
    id_col: str = 'personid',
    icd_pheno_col: str = 'IcdPheno',
    lab_pheno_col: str = 'LabPhenotype'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Combine ICD and lab phenotype assignments by outer join.

    Keeps both IcdPheno and PersonPhenotype as separate columns.
    No consensus — business rules are applied downstream in R targets.

    The SCD cohort is extracted using OR logic:
        PersonPhenotype in SCD_LAB_PHENOTYPES OR IcdPheno in SCD_ICD_PHENOTYPES

    Parameters
    ----------
    icd_df : pd.DataFrame
        ICD phenotype results. Must have id_col and icd_pheno_col.
    lab_df : pd.DataFrame
        Lab phenotype results. Must have id_col and lab_pheno_col.
    id_col : str
        Person identifier column name.
    icd_pheno_col : str
        Column with ICD phenotype labels.
    lab_pheno_col : str
        Column with lab phenotype labels (will be renamed to PersonPhenotype).

    Returns
    -------
    tuple of (combined_df, scd_cohort_df)
        combined_df: All persons with ICD and/or lab phenotypes.
            Columns: personid, IcdPheno, PersonPhenotype, [HgbSMax, HgbAMax, TfxPercent]
        scd_cohort_df: Filtered to SCD cohort (OR logic).
    """
    logger.info("Combining ICD and lab phenotype assignments")

    # Select relevant columns from ICD
    icd_cols = [id_col, icd_pheno_col]
    icd_subset = icd_df[icd_cols].copy()

    # Select relevant columns from Lab, rename to PersonPhenotype
    lab_cols = [id_col, lab_pheno_col]
    for extra in ['HgbSMax', 'HgbAMax', 'TfxPercent']:
        if extra in lab_df.columns:
            lab_cols.append(extra)
    lab_subset = lab_df[[c for c in lab_cols if c in lab_df.columns]].copy()
    lab_subset = lab_subset.rename(columns={lab_pheno_col: 'PersonPhenotype'})

    # Outer join — keep both columns, lose no patients
    combined = icd_subset.merge(lab_subset, on=id_col, how='outer')

    n_icd_only = combined[icd_pheno_col].notna().sum() - (combined[icd_pheno_col].notna() & combined['PersonPhenotype'].notna()).sum()
    n_lab_only = combined['PersonPhenotype'].notna().sum() - (combined[icd_pheno_col].notna() & combined['PersonPhenotype'].notna()).sum()
    n_both = (combined[icd_pheno_col].notna() & combined['PersonPhenotype'].notna()).sum()

    logger.info(
        f"Combined: {len(combined)} persons "
        f"(ICD only: {n_icd_only}, Lab only: {n_lab_only}, Both: {n_both})"
    )

    # Log distributions
    logger.info(f"IcdPheno distribution:\n{combined[icd_pheno_col].value_counts().to_string()}")
    logger.info(f"PersonPhenotype distribution:\n{combined['PersonPhenotype'].value_counts().to_string()}")

    # Extract SCD cohort using OR logic (matches reference 050 notebook)
    scd_mask = (
        combined['PersonPhenotype'].isin(SCD_LAB_PHENOTYPES)
        | combined[icd_pheno_col].isin(SCD_ICD_PHENOTYPES)
    )
    scd_cohort = combined[scd_mask].copy()

    logger.info(f"SCD cohort: {len(scd_cohort)} persons")

    return combined, scd_cohort
