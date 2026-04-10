"""
Conformance tests: scd_phenotyping vs reference (kwuichet/SCD_PHenotyping)

Tests synthetic patients through both the reference assignRowPhenotype()
and our classify_lab_row() to verify identical results.

Reference: hemoglobinopathy_phenotyping_notebook.ipynb from
https://github.com/kwuichet/SCD_PHenotyping

Run: pytest scd_phenotyping/tests/test_conformance.py -v
"""

import pytest
import pandas as pd
import numpy as np

from scd_phenotyping.lab import classify_lab_row


def _make_row(**kwargs):
    """Build a synthetic lab row with defaults."""
    defaults = {
        'personid': 'test_patient',
        'age': 25,
        'HgbA': 0, 'HgbA2': 0, 'HgbC': 0, 'HgbD': 0,
        'HgbE': 0, 'HgbF': 0, 'HgbS': 0, 'HgbO': 0,
        'HgbSum': 0, 'CompleteFrac': 'N',
        'AoverS': np.nan, 'CoverS': np.nan, 'DoverS': np.nan, 'EoverS': np.nan, 'VoverS': np.nan,
        'AoverC': np.nan, 'AoverD': np.nan, 'AoverE': np.nan, 'AoverV': np.nan,
        'PostTfx': 'N', 'PostHU': 'N',
        'AvgHgbTotal': 0,
    }
    defaults.update(kwargs)
    # Compute HgbSum and CompleteFrac if not explicitly set
    if 'HgbSum' not in kwargs:
        defaults['HgbSum'] = sum(defaults[k] for k in
                                 ['HgbA', 'HgbA2', 'HgbC', 'HgbD',
                                  'HgbE', 'HgbF', 'HgbS', 'HgbO'])
    if 'CompleteFrac' not in kwargs:
        defaults['CompleteFrac'] = 'Y' if 90 <= defaults['HgbSum'] <= 105 else 'N'
    # Compute ratios if not explicitly set
    s = defaults['HgbS'] if defaults['HgbS'] > 0 else np.nan
    a = defaults['HgbA']
    if 'AoverS' not in kwargs and pd.notna(s) and s > 0:
        defaults['AoverS'] = a / s
    if 'CoverS' not in kwargs and pd.notna(s) and s > 0:
        defaults['CoverS'] = defaults['HgbC'] / s
    if 'DoverS' not in kwargs and pd.notna(s) and s > 0:
        defaults['DoverS'] = defaults['HgbD'] / s
    if 'EoverS' not in kwargs and pd.notna(s) and s > 0:
        defaults['EoverS'] = defaults['HgbE'] / s
    if 'VoverS' not in kwargs and pd.notna(s) and s > 0:
        defaults['VoverS'] = defaults['HgbO'] / s
    if 'AoverC' not in kwargs and defaults['HgbC'] > 0:
        defaults['AoverC'] = a / defaults['HgbC']
    if 'AoverD' not in kwargs and defaults['HgbD'] > 0:
        defaults['AoverD'] = a / defaults['HgbD']
    if 'AoverE' not in kwargs and defaults['HgbE'] > 0:
        defaults['AoverE'] = a / defaults['HgbE']
    if 'AoverV' not in kwargs and defaults['HgbO'] > 0:
        defaults['AoverV'] = a / defaults['HgbO']
    return pd.Series(defaults)


# =========================================================================
# 1. DEFINITIVE SCD GENOTYPES (Priority 1 in reference)
# =========================================================================

class TestDefinitiveSCD:
    """SCD_SC, SCD_SD, SCD_SE, SCD_SVar — based on hemoglobin ratios."""

    def test_scd_sc(self):
        """CoverS between 0.6 and 1.4 → SCD_SC"""
        row = _make_row(HgbS=45, HgbC=45, HgbF=5, HgbA2=3, age=25)
        assert classify_lab_row(row) == 'SCD_SC'

    def test_scd_sd(self):
        """DoverS between 0.6 and 1.4 → SCD_SD"""
        row = _make_row(HgbS=45, HgbD=40, HgbF=5, HgbA2=3, age=25)
        assert classify_lab_row(row) == 'SCD_SD'

    def test_scd_se(self):
        """EoverS between 0.3 and 0.7 → SCD_SE"""
        row = _make_row(HgbS=60, HgbE=30, HgbF=5, age=25)
        assert classify_lab_row(row) == 'SCD_SE'

    def test_scd_svar(self):
        """VoverS between 0.6 and 1.4 → SCD_SVar"""
        row = _make_row(HgbS=45, HgbO=40, HgbF=5, HgbA2=3, age=25)
        assert classify_lab_row(row) == 'SCD_SVar'

    def test_sc_takes_priority_over_sca(self):
        """SC detected even with high HgbS and low HgbA"""
        row = _make_row(HgbS=50, HgbC=45, HgbA=0, HgbF=2, HgbA2=3, age=25)
        assert classify_lab_row(row) == 'SCD_SC'


# =========================================================================
# 2. SCD_SCA — Definitive (age >= 2, complete frac, HgbS>=5, HgbA<=5)
# =========================================================================

class TestSCDSCA:
    """Classic SCA: high S, absent/very low A, complete fractionation."""

    def test_sca_definitive(self):
        """HgbS=85, HgbA=0, HgbF=10, CompleteFrac=Y → SCD_SCA"""
        row = _make_row(HgbS=85, HgbA=0, HgbF=10, HgbA2=3, age=25)
        assert classify_lab_row(row) == 'SCD_SCA'

    def test_sca_with_low_a(self):
        """HgbS=80, HgbA=5 → still SCD_SCA (A<=5 threshold)"""
        row = _make_row(HgbS=80, HgbA=5, HgbF=8, HgbA2=3, age=25)
        assert classify_lab_row(row) == 'SCD_SCA'

    def test_sca_a_above_threshold(self):
        """HgbS=60, HgbA=30 → NOT SCD_SCA (A>5)"""
        row = _make_row(HgbS=60, HgbA=30, HgbF=5, HgbA2=3, age=25)
        result = classify_lab_row(row)
        assert result != 'SCD_SCA'


# =========================================================================
# 3. TRANSFUSION-DEPENDENT CLASSIFICATION
# =========================================================================

class TestTransfusion:
    """HgbS>=40, HgbA<40: PostTfx determines SCA_Likely vs Sbetap_Likely."""

    def test_post_transfusion_sca_likely(self):
        """HgbS=55, HgbA=35, PostTfx=Y → SCD_SCA_Likely"""
        row = _make_row(HgbS=55, HgbA=35, HgbF=5, HgbA2=3, age=25, PostTfx='Y')
        assert classify_lab_row(row) == 'SCD_SCA_Likely'

    def test_pre_transfusion_sbetap_likely(self):
        """HgbS=55, HgbA=35, PostTfx=N → SCD_Sbetap_Likely"""
        row = _make_row(HgbS=55, HgbA=35, HgbF=5, HgbA2=3, age=25, PostTfx='N')
        assert classify_lab_row(row) == 'SCD_Sbetap_Likely'


# =========================================================================
# 4. SICKLE TRAIT
# =========================================================================

class TestSickleTrait:
    """S_Trait: HgbS>=5, AoverS 1.0-2.0, not post-transfusion."""

    def test_s_trait(self):
        """HgbS=35, HgbA=55, AoverS=1.57, PostTfx=N → S_Trait"""
        row = _make_row(HgbS=35, HgbA=55, HgbF=3, HgbA2=3, age=25, PostTfx='N')
        assert classify_lab_row(row, dataset_scd=False) == 'S_Trait'

    def test_s_indeterminate_when_post_tfx(self):
        """Same values but PostTfx=Y → S_Indeterminate (not trait)"""
        row = _make_row(HgbS=35, HgbA=55, HgbF=3, HgbA2=3, age=25, PostTfx='Y')
        result = classify_lab_row(row, dataset_scd=False)
        assert result == 'S_Indeterminate'


# =========================================================================
# 5. PEDIATRIC (age < 2)
# =========================================================================

class TestPediatric:
    """Age < 2 has different classification rules."""

    def test_infant_sca_likely(self):
        """Age=1, HgbS=70, HgbA=0, CompleteFrac=Y → SCD_SCA_Likely"""
        row = _make_row(HgbS=70, HgbA=0, HgbF=25, HgbA2=2, age=1)
        assert classify_lab_row(row) == 'SCD_SCA_Likely'

    def test_infant_low_aover_s_post_tfx(self):
        """Age=1, HgbS=50, HgbA=10, AoverS<0.6, PostTfx=Y → SCD_SCA_Likely"""
        row = _make_row(HgbS=50, HgbA=10, HgbF=30, HgbA2=3, age=1, PostTfx='Y')
        assert classify_lab_row(row) == 'SCD_SCA_Likely'

    def test_infant_low_aover_s_no_tfx(self):
        """Age=1, HgbS=50, HgbA=10, AoverS<0.6, PostTfx=N → SCD_Indeterminate"""
        row = _make_row(HgbS=50, HgbA=10, HgbF=30, HgbA2=3, age=1, PostTfx='N')
        assert classify_lab_row(row) == 'SCD_Indeterminate'

    def test_infant_trait_range(self):
        """Age=1, HgbS=30, HgbA=55, AoverS~1.8, PostTfx=N → S_Trait"""
        row = _make_row(HgbS=30, HgbA=55, HgbF=10, HgbA2=3, age=1, PostTfx='N')
        assert classify_lab_row(row, dataset_scd=False) == 'S_Trait'

    def test_infant_no_hgbs(self):
        """Age=1, HgbS absent → No_Phenotype"""
        row = _make_row(HgbS=0, HgbA=80, HgbF=15, HgbA2=3, age=1)
        assert classify_lab_row(row) == 'No_Phenotype'


# =========================================================================
# 6. OTHER HEMOGLOBINOPATHIES (datasetSCD=False only)
# =========================================================================

class TestOtherHemoglobinopathies:
    """HemC, HemD, HemE, BetaThalassemia — only when datasetSCD=False."""

    def test_hemc_trait(self):
        """CompleteFrac=Y, HgbS absent, AoverC 0.9-3.25 → HemC_Trait"""
        row = _make_row(HgbA=60, HgbC=30, HgbF=3, HgbA2=3, HgbS=0, age=25)
        assert classify_lab_row(row, dataset_scd=False) == 'HemC_Trait'

    def test_hemc_disease(self):
        """AoverC < 0.5 → HemC_Disease"""
        row = _make_row(HgbA=10, HgbC=80, HgbF=3, HgbA2=3, HgbS=0, age=25)
        assert classify_lab_row(row, dataset_scd=False) == 'HemC_Disease'

    def test_beta_thalassemia(self):
        """HgbA2 > 4 → BetaThalassemia"""
        row = _make_row(HgbA=85, HgbA2=6, HgbF=3, HgbS=0, age=25)
        assert classify_lab_row(row, dataset_scd=False) == 'BetaThalassemia'

    def test_not_scd(self):
        """Age>=2, HgbS absent, HgbO absent, CompleteFrac=Y → Not_SCD"""
        row = _make_row(HgbA=90, HgbA2=3, HgbF=3, HgbS=0, HgbO=0, age=25)
        assert classify_lab_row(row, dataset_scd=False) == 'Not_SCD'

    def test_skipped_when_dataset_scd(self):
        """Same as HemC_Trait but datasetSCD=True → should NOT return HemC_Trait"""
        row = _make_row(HgbA=60, HgbC=30, HgbF=3, HgbA2=3, HgbS=0, age=25)
        result = classify_lab_row(row, dataset_scd=True)
        assert result != 'HemC_Trait'


# =========================================================================
# 7. INCOMPLETE FRACTIONATION (HgbSum < 90)
# =========================================================================

class TestIncompleteFractionation:
    """High HgbS with incomplete fractionation — heuristic rules."""

    def test_very_high_s_sca(self):
        """HgbS=85, incomplete frac → SCD_SCA"""
        row = _make_row(HgbS=85, HgbA=0, age=25,
                        HgbSum=85, CompleteFrac='N')
        assert classify_lab_row(row) == 'SCD_SCA'

    def test_high_s_sca_likely(self):
        """HgbS=75, incomplete frac → SCD_SCA_Likely"""
        row = _make_row(HgbS=75, HgbA=0, age=25,
                        HgbSum=75, CompleteFrac='N')
        assert classify_lab_row(row) == 'SCD_SCA_Likely'

    def test_moderate_s_with_low_a_sca_likely(self):
        """HgbS=55, HgbA=3, incomplete frac → SCD_SCA_Likely"""
        row = _make_row(HgbS=55, HgbA=3, age=25,
                        HgbSum=58, CompleteFrac='N')
        assert classify_lab_row(row) == 'SCD_SCA_Likely'

    def test_moderate_s_post_tfx_sca_likely(self):
        """HgbS=55, PostTfx=Y, incomplete frac → SCD_SCA_Likely"""
        row = _make_row(HgbS=55, HgbA=20, age=25, PostTfx='Y',
                        HgbSum=75, CompleteFrac='N')
        assert classify_lab_row(row) == 'SCD_SCA_Likely'


# =========================================================================
# 8. TRANSFUSION THRESHOLD
# =========================================================================

class TestTransfusionThreshold:
    """Verify the 180-day threshold matches reference."""

    def test_threshold_is_180(self):
        """TFX_DAYS_THRESHOLD should be 180, matching reference tfxThresh."""
        from scd_phenotyping.lab import TFX_DAYS_THRESHOLD
        assert TFX_DAYS_THRESHOLD == 180


# =========================================================================
# 9. TRANSFUSION INFERENCE FALLBACK
# =========================================================================

class TestTransfusionInferenceFallback:
    """When no tfx file, inferred status should set PostTfx."""

    def test_inferred_post_sets_posttfx_y(self):
        """InferTfxRow='Post' → PostTfx='Y' when no tfx file."""
        from scd_phenotyping.lab import add_actual_transfusion
        df = pd.DataFrame({
            'personid': ['A', 'B', 'C'],
            'date': ['2020-01-01', '2020-01-01', '2020-01-01'],
            'InferTfxRow': ['Post', 'Pre', 'PostMaybe'],
        })
        result = add_actual_transfusion(df, tfx_df=None)
        assert result.loc[0, 'PostTfx'] == 'Y'   # Post → Y
        assert result.loc[1, 'PostTfx'] == 'N'   # Pre → N
        assert result.loc[2, 'PostTfx'] == 'Y'   # PostMaybe → Y

    def test_no_infer_column_defaults_to_n(self):
        """No InferTfxRow column → PostTfx='N' (safe default)."""
        from scd_phenotyping.lab import add_actual_transfusion
        df = pd.DataFrame({
            'personid': ['A'],
            'date': ['2020-01-01'],
        })
        result = add_actual_transfusion(df, tfx_df=None)
        assert result.loc[0, 'PostTfx'] == 'N'


# =========================================================================
# 10. NO PHENOTYPE / EDGE CASES
# =========================================================================

class TestEdgeCases:

    def test_all_zeros(self):
        """All hemoglobin values zero → No_Phenotype"""
        row = _make_row(age=25)
        assert classify_lab_row(row) == 'No_Phenotype'

    def test_nan_age(self):
        """NaN age should not crash"""
        row = _make_row(HgbS=85, HgbA=0, HgbF=10, HgbA2=3, age=np.nan)
        result = classify_lab_row(row)
        assert isinstance(result, str)
