"""
Tests for combined ICD + Lab phenotyping.
"""

import pandas as pd
import pytest

from scd_phenotyping.combined import run_combined_phenotyping, SCD_LAB_PHENOTYPES


class TestRunCombinedPhenotyping:
    def test_lab_takes_precedence_for_scd(self):
        """Lab phenotype provides more specific SCD subtype."""
        icd_df = pd.DataFrame({
            'personid': [1, 2],
            'IcdPheno': ['SCD', 'SCD'],
        })
        lab_df = pd.DataFrame({
            'personid': [1, 2],
            'LabPhenotype': ['SCD_SCA', 'SCD_SC'],
        })
        combined, scd = run_combined_phenotyping(icd_df, lab_df)
        assert len(combined) == 2
        assert combined[combined['personid'] == 1]['PersonPhenotype'].iloc[0] == 'SCD_SCA'
        assert combined[combined['personid'] == 2]['PersonPhenotype'].iloc[0] == 'SCD_SC'
        assert len(scd) == 2

    def test_icd_only_fallback(self):
        """ICD phenotype used when no lab data."""
        icd_df = pd.DataFrame({
            'personid': [1],
            'IcdPheno': ['SCD'],
        })
        lab_df = pd.DataFrame({
            'personid': [2],
            'LabPhenotype': ['SCD_SCA'],
        })
        combined, scd = run_combined_phenotyping(icd_df, lab_df)
        assert len(combined) == 2
        p1 = combined[combined['personid'] == 1]
        # run_combined_phenotyping keeps both IcdPheno and PersonPhenotype
        # separate; there is no ConsensusPhenotype or PhenotypeSource.
        # ICD-only persons get IcdPheno set, PersonPhenotype null.
        assert p1['IcdPheno'].iloc[0] == 'SCD'
        assert pd.isna(p1['PersonPhenotype'].iloc[0])

    def test_scd_cohort_extraction(self):
        """SCD cohort includes both ICD=SCD and Lab=SCD_*."""
        icd_df = pd.DataFrame({
            'personid': [1, 2, 3],
            'IcdPheno': ['SCD', 'TRAIT', 'UNK'],
        })
        lab_df = pd.DataFrame({
            'personid': [2, 3],
            'LabPhenotype': ['S_Trait', 'SCD_SCA_Likely'],
        })
        combined, scd = run_combined_phenotyping(icd_df, lab_df)
        # Person 1: ICD=SCD → in cohort
        # Person 2: ICD=TRAIT, Lab=S_Trait → NOT in cohort
        # Person 3: ICD=UNK, Lab=SCD_SCA_Likely → in cohort
        scd_ids = set(scd['personid'])
        assert 1 in scd_ids
        assert 2 not in scd_ids
        assert 3 in scd_ids
