"""
Combined ICD + Lab Phenotyping
==============================

Merges ICD-based and lab-based phenotype assignments into a consensus
phenotype per person. The lab phenotype takes precedence for SCD subtyping
when available; ICD phenotype confirms SCD presence.

Reference: 050-Hemoglobinopath_Phenotyping-Run.ipynb (Stage 3: Combine)
"""

import logging
from typing import Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Lab phenotypes that indicate SCD (used for final cohort extraction)
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

# ICD phenotypes that indicate SCD
SCD_ICD_PHENOTYPES = ['SCD', 'SCDX']


def run_combined_phenotyping(
    icd_df: pd.DataFrame,
    lab_df: pd.DataFrame,
    id_col: str = 'personid',
    icd_pheno_col: str = 'IcdPheno',
    lab_pheno_col: str = 'LabPhenotype'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Combine ICD and lab phenotype assignments into a consensus phenotype.

    The combination logic:
    1. Outer join ICD and Lab results on person ID
    2. Assign consensus phenotype:
       - If lab phenotype is SCD-specific → use lab phenotype (more precise)
       - If only ICD phenotype available → use ICD phenotype
       - If only lab phenotype available → use lab phenotype
       - If both present but disagree → flag for review
    3. Extract final SCD cohort (either ICD=SCD or Lab in SCD_LAB_PHENOTYPES)

    Parameters
    ----------
    icd_df : pd.DataFrame
        ICD phenotype results (from run_icd_phenotyping). Must have id_col
        and icd_pheno_col columns.
    lab_df : pd.DataFrame
        Lab phenotype results (from run_lab_phenotyping or aggregate_lab_phenotypes).
        Must have id_col and lab_pheno_col columns.
    id_col : str
        Person identifier column name.
    icd_pheno_col : str
        Column with ICD phenotype labels.
    lab_pheno_col : str
        Column with lab phenotype labels.

    Returns
    -------
    tuple of (combined_df, scd_cohort_df)
        combined_df: All persons with ICD and/or lab phenotypes, plus consensus.
        scd_cohort_df: Filtered to SCD cohort only.
    """
    logger.info("Combining ICD and lab phenotype assignments")

    # Select relevant columns
    icd_cols = [id_col, icd_pheno_col]
    lab_cols = [id_col, lab_pheno_col]

    # Add extra lab columns if available
    for extra in ['HgbSMax', 'HgbAMax', 'TfxPercent']:
        if extra in lab_df.columns:
            lab_cols.append(extra)

    icd_subset = icd_df[icd_cols].copy()
    lab_subset = lab_df[[c for c in lab_cols if c in lab_df.columns]].copy()

    # Outer join
    combined = icd_subset.merge(lab_subset, on=id_col, how='outer')

    logger.info(
        f"Combined: {len(combined)} persons "
        f"(ICD only: {combined[icd_pheno_col].notna().sum() - combined[lab_pheno_col].notna().sum()}, "
        f"Lab only: {combined[lab_pheno_col].notna().sum() - combined[icd_pheno_col].notna().sum()}, "
        f"Both: {(combined[icd_pheno_col].notna() & combined[lab_pheno_col].notna()).sum()})"
    )

    # Assign consensus phenotype
    combined['ConsensusPhenotype'] = combined.apply(
        lambda r: _assign_consensus(r, icd_pheno_col, lab_pheno_col),
        axis=1
    )

    # Flag disagreements
    combined['PhenotypeSource'] = combined.apply(
        lambda r: _phenotype_source(r, icd_pheno_col, lab_pheno_col),
        axis=1
    )

    consensus_counts = combined['ConsensusPhenotype'].value_counts()
    logger.info(f"Consensus phenotype distribution:\n{consensus_counts.to_string()}")

    # Extract SCD cohort
    scd_mask = (
        combined[icd_pheno_col].isin(SCD_ICD_PHENOTYPES)
        | combined[lab_pheno_col].isin(SCD_LAB_PHENOTYPES)
    )
    scd_cohort = combined[scd_mask].copy()

    logger.info(f"SCD cohort: {len(scd_cohort)} persons")

    return combined, scd_cohort


def _assign_consensus(row, icd_col, lab_col):
    """Assign consensus phenotype from ICD + Lab."""
    icd = row.get(icd_col)
    lab = row.get(lab_col)

    icd_valid = pd.notna(icd) and icd not in ('UNK', 'No_Phenotype')
    lab_valid = pd.notna(lab) and lab not in ('No_Phenotype',)

    # Lab phenotype is more specific for SCD subtyping
    if lab_valid and lab in SCD_LAB_PHENOTYPES:
        return lab

    # Non-SCD lab phenotypes
    if lab_valid and not icd_valid:
        return lab

    # ICD only
    if icd_valid and not lab_valid:
        return icd

    # Both valid — lab takes precedence for SCD subtypes
    if lab_valid and icd_valid:
        if lab in SCD_LAB_PHENOTYPES:
            return lab
        # ICD says SCD but lab disagrees — flag
        if icd in SCD_ICD_PHENOTYPES and lab not in SCD_LAB_PHENOTYPES:
            return f'{icd}_ICD_only'
        return lab

    return 'Unclassified'


def _phenotype_source(row, icd_col, lab_col):
    """Determine the source of the consensus phenotype."""
    icd = row.get(icd_col)
    lab = row.get(lab_col)

    icd_valid = pd.notna(icd) and icd not in ('UNK', 'No_Phenotype')
    lab_valid = pd.notna(lab) and lab not in ('No_Phenotype',)

    if icd_valid and lab_valid:
        return 'Both'
    if icd_valid:
        return 'ICD_only'
    if lab_valid:
        return 'Lab_only'
    return 'None'
