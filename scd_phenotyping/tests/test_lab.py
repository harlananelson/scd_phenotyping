"""
Tests for lab-based phenotyping.

Uses synthetic data — no Git LFS dependencies.
"""

import pandas as pd
import numpy as np
import pytest

from scd_phenotyping.lab import (
    extract_numeric,
    calculate_ratios,
    classify_lab_row,
    aggregate_lab_phenotypes,
    infer_transfusion_status,
    pivot_lab_data,
    run_lab_phenotyping,
)


# ---------------------------------------------------------------------------
# extract_numeric
# ---------------------------------------------------------------------------

class TestExtractNumeric:
    def test_float(self):
        assert extract_numeric(45.2) == 45.2

    def test_int(self):
        assert extract_numeric(90) == 90.0

    def test_gt_string(self):
        assert extract_numeric(">90") == 90.0

    def test_lt_string(self):
        assert extract_numeric("<1") == 1.0

    def test_percent_string(self):
        assert extract_numeric("45.2%") == 45.2

    def test_nan(self):
        assert extract_numeric(np.nan) is None

    def test_none(self):
        assert extract_numeric(None) is None

    def test_no_number(self):
        assert extract_numeric("absent") is None


# ---------------------------------------------------------------------------
# calculate_ratios
# ---------------------------------------------------------------------------

class TestCalculateRatios:
    def test_basic_percentages(self):
        df = pd.DataFrame({
            'HgbA': [10.0],
            'HgbS': [80.0],
            'HgbF': [5.0],
            'HgbC': [0.0],
        })
        result = calculate_ratios(df)
        assert 'HgbSum' in result.columns
        assert result['HgbSum'].iloc[0] == 95.0
        assert result['CompleteFrac'].iloc[0] == 'Y'  # 90-105

    def test_incomplete_fractionation(self):
        df = pd.DataFrame({
            'HgbA': [5.0],
            'HgbS': [40.0],
            'HgbF': [2.0],
            'HgbC': [0.0],
        })
        result = calculate_ratios(df)
        assert result['CompleteFrac'].iloc[0] == 'N'  # sum=47 < 90

    def test_ratios_computed(self):
        df = pd.DataFrame({
            'HgbA': [20.0],
            'HgbS': [70.0],
            'HgbF': [5.0],
            'HgbC': [0.0],
        })
        result = calculate_ratios(df)
        assert 'AoverS' in result.columns
        assert abs(result['AoverS'].iloc[0] - 20.0 / 70.0) < 0.001

    def test_zero_denominator_ratio(self):
        df = pd.DataFrame({
            'HgbA': [95.0],
            'HgbS': [0.0],
            'HgbF': [2.0],
            'HgbC': [0.0],
        })
        result = calculate_ratios(df)
        assert pd.isna(result['AoverS'].iloc[0])  # can't divide by 0

    def test_optional_subs_included(self):
        df = pd.DataFrame({
            'HgbA': [40.0],
            'HgbS': [40.0],
            'HgbF': [5.0],
            'HgbC': [0.0],
            'HgbD': [10.0],
        })
        result = calculate_ratios(df)
        assert result['HgbSum'].iloc[0] == 95.0
        assert 'DoverS' in result.columns


# ---------------------------------------------------------------------------
# classify_lab_row — row-level classification
# ---------------------------------------------------------------------------

class TestClassifyLabRow:
    def _make_row(self, **kwargs):
        """Helper to create a row with defaults."""
        defaults = {
            'HgbA': 0, 'HgbS': 0, 'HgbF': 0, 'HgbC': 0,
            'HgbD': 0, 'HgbE': 0, 'HgbO': 0, 'HgbA2': 0,
            'HgbSum': 0, 'CompleteFrac': 'Y',
            'PostTfx': 'N', 'PostHU': 'N',
            'age': 25,
            'AoverS': None, 'CoverS': None, 'DoverS': None,
            'EoverS': None, 'VoverS': None,
            'AoverC': None, 'AoverD': None, 'AoverE': None, 'AoverV': None,
        }
        defaults.update(kwargs)
        # Auto-calculate ratios if not specified
        if defaults['HgbS'] > 0 and defaults['AoverS'] is None:
            defaults['AoverS'] = defaults['HgbA'] / defaults['HgbS']
        if defaults['HgbS'] > 0 and defaults['CoverS'] is None:
            defaults['CoverS'] = defaults['HgbC'] / defaults['HgbS']
        if defaults['HgbS'] > 0 and defaults['DoverS'] is None and defaults['HgbD'] > 0:
            defaults['DoverS'] = defaults['HgbD'] / defaults['HgbS']
        if defaults['HgbC'] > 0 and defaults['AoverC'] is None:
            defaults['AoverC'] = defaults['HgbA'] / defaults['HgbC']
        defaults['HgbSum'] = sum(
            defaults[k] for k in ['HgbA', 'HgbS', 'HgbF', 'HgbC', 'HgbD', 'HgbE', 'HgbO']
        )
        if 90 <= defaults['HgbSum'] <= 105:
            defaults['CompleteFrac'] = 'Y'
        else:
            defaults['CompleteFrac'] = 'N'
        return pd.Series(defaults)

    def test_scd_sca_definitive(self):
        """High S, no A, no O → SCD_SCA."""
        row = self._make_row(HgbS=85, HgbF=10, HgbA=0)
        assert classify_lab_row(row) == 'SCD_SCA'

    def test_scd_sc_compound(self):
        """HgbC/HgbS ratio ~1.0 → SCD_SC."""
        row = self._make_row(HgbS=45, HgbC=45, HgbF=5)
        assert classify_lab_row(row) == 'SCD_SC'

    def test_scd_sd_compound(self):
        """HgbD/HgbS ratio ~1.0 → SCD_SD."""
        row = self._make_row(HgbS=45, HgbD=45, HgbF=5)
        assert classify_lab_row(row) == 'SCD_SD'

    def test_scd_sbetap_likely(self):
        """S >= 40, A < 40, not post-transfusion → SCD_Sbetap_Likely."""
        row = self._make_row(HgbS=55, HgbA=35, HgbF=5)
        assert classify_lab_row(row) == 'SCD_Sbetap_Likely'

    def test_scd_sca_likely_post_tfx(self):
        """S >= 40, A < 40, post-transfusion → SCD_SCA_Likely."""
        row = self._make_row(HgbS=55, HgbA=35, HgbF=5, PostTfx='Y')
        assert classify_lab_row(row) == 'SCD_SCA_Likely'

    def test_s_trait(self):
        """A > S, AoverS in [1.0, 2.0], not post-tfx → S_Trait."""
        row = self._make_row(HgbS=35, HgbA=55, HgbF=5)
        assert classify_lab_row(row) == 'S_Trait'

    def test_pediatric_sca_likely(self):
        """Age < 2, S >= 5, A <= 1 → SCD_SCA_Likely."""
        row = self._make_row(HgbS=80, HgbA=0, HgbF=15, age=1)
        assert classify_lab_row(row) == 'SCD_SCA_Likely'

    def test_pediatric_trait(self):
        """Age < 2, AoverS in [0.6, 2.0], not post-tfx → S_Trait."""
        row = self._make_row(HgbS=30, HgbA=40, HgbF=25, age=1)
        assert classify_lab_row(row) == 'S_Trait'

    def test_no_phenotype_no_s(self):
        """No HgbS → No_Phenotype (age >= 2)."""
        row = self._make_row(HgbA=95, HgbF=2)
        assert classify_lab_row(row) == 'No_Phenotype'

    def test_incomplete_high_s(self):
        """Incomplete fractionation, HgbS >= 80 → SCD_SCA."""
        row = self._make_row(HgbS=82, HgbF=3)  # sum < 90
        assert classify_lab_row(row) == 'SCD_SCA'

    def test_incomplete_moderate_s(self):
        """Incomplete fractionation, HgbS >= 70 → SCD_SCA_Likely."""
        row = self._make_row(HgbS=72, HgbF=3)  # sum < 90
        assert classify_lab_row(row) == 'SCD_SCA_Likely'

    def test_spfh(self):
        """High F, high S, high total Hgb, not on HU → SCD_SPFH."""
        row = self._make_row(HgbS=50, HgbF=25, HgbA=20)
        result = classify_lab_row(row, run_spfh=True, avg_hgb_total=13.0)
        assert result == 'SCD_SPFH'

    def test_spfh_blocked_by_hu(self):
        """Same as SPFH but on HU → NOT SPFH."""
        row = self._make_row(HgbS=50, HgbF=25, HgbA=20, PostHU='Y')
        result = classify_lab_row(row, run_spfh=True, avg_hgb_total=13.0)
        assert result != 'SCD_SPFH'

    def test_non_scd_hemc_trait(self):
        """Non-SCD dataset, low S, AoverC in trait range → HemC_Trait."""
        row = self._make_row(HgbA=60, HgbC=30, HgbS=0, HgbF=5)
        result = classify_lab_row(row, dataset_scd=False)
        assert result == 'HemC_Trait'

    def test_beta_thalassemia(self):
        """HgbA2 > 4 → BetaThalassemia."""
        row = self._make_row(HgbA=85, HgbS=0, HgbF=3, HgbA2=7)
        result = classify_lab_row(row, dataset_scd=False)
        assert result == 'BetaThalassemia'


# ---------------------------------------------------------------------------
# aggregate_lab_phenotypes — person-level priority cascade
# ---------------------------------------------------------------------------

class TestAggregateLabPhenotypes:
    def test_sca_takes_precedence(self):
        """SCD_SCA in any row → person gets SCD_SCA."""
        df = pd.DataFrame({
            'personid': [1, 1, 1],
            'LabPhenotype': ['S_Indeterminate', 'SCD_SCA', 'S_Trait'],
            'HgbS': [30, 85, 35],
            'PostTfx': ['N', 'N', 'N'],
        })
        result = aggregate_lab_phenotypes(df)
        assert result.loc[0, 'LabPhenotype'] == 'SCD_SCA'

    def test_compound_takes_precedence(self):
        """SCD_SC compound → takes top priority."""
        df = pd.DataFrame({
            'personid': [1, 1],
            'LabPhenotype': ['SCD_SC', 'S_Trait'],
            'HgbS': [45, 35],
            'PostTfx': ['N', 'N'],
        })
        result = aggregate_lab_phenotypes(df)
        assert result.loc[0, 'LabPhenotype'] == 'SCD_SC'

    def test_tfx_disambiguation(self):
        """Sbetap + SCA_Likely with high TfxPercent → SCD_SCA_Likely."""
        df = pd.DataFrame({
            'personid': [1, 1, 1, 1],
            'LabPhenotype': ['SCD_Sbetap_Likely', 'SCD_SCA_Likely', 'SCD_SCA_Likely', 'SCD_SCA_Likely'],
            'HgbS': [55, 55, 55, 55],
            'PostTfx': ['N', 'Y', 'Y', 'Y'],
        })
        result = aggregate_lab_phenotypes(df)
        # 75% post-tfx → SCA_Likely wins
        assert result.loc[0, 'LabPhenotype'] == 'SCD_SCA_Likely'

    def test_multiple_persons(self):
        """Multiple persons aggregated correctly."""
        df = pd.DataFrame({
            'personid': [1, 1, 2, 2],
            'LabPhenotype': ['SCD_SCA', 'SCD_SCA', 'S_Trait', 'S_Trait'],
            'HgbS': [85, 82, 35, 33],
            'PostTfx': ['N', 'N', 'N', 'N'],
        })
        result = aggregate_lab_phenotypes(df)
        assert len(result) == 2
        assert result[result['personid'] == 1]['LabPhenotype'].iloc[0] == 'SCD_SCA'
        assert result[result['personid'] == 2]['LabPhenotype'].iloc[0] == 'S_Trait'


# ---------------------------------------------------------------------------
# infer_transfusion_status
# ---------------------------------------------------------------------------

class TestInferTransfusion:
    def test_stable_no_transfusion(self):
        """Small HgbA/S variation → no transfusion inferred."""
        df = pd.DataFrame({
            'personid': [1, 1, 1],
            'HgbA': [2, 3, 2],
            'HgbS': [80, 82, 81],
        })
        result = infer_transfusion_status(df)
        assert all(result['InferTfxPerson'] == 'No')

    def test_high_variation_transfusion(self):
        """Large HgbA range → transfusion inferred."""
        df = pd.DataFrame({
            'personid': [1, 1, 1],
            'HgbA': [2, 25, 3],
            'HgbS': [80, 55, 78],
        })
        result = infer_transfusion_status(df)
        assert all(result['InferTfxPerson'] == 'Yes')
        # Row with elevated HgbA should be Post
        assert result.loc[1, 'InferTfxRow'] == 'Post'


# ---------------------------------------------------------------------------
# pivot_lab_data
# ---------------------------------------------------------------------------

class TestPivotLabData:
    def test_basic_pivot(self):
        df = pd.DataFrame({
            'personid': [1, 1, 1, 1],
            'date': ['2024-01-01'] * 4,
            'age': [25] * 4,
            'hgbType': ['HgbA', 'HgbS', 'HgbF', 'HgbC'],
            'value': [5, 80, 10, 0],
        })
        result = pivot_lab_data(df)
        assert len(result) == 1
        assert result['HgbS'].iloc[0] == 80.0

    def test_string_values_extracted(self):
        df = pd.DataFrame({
            'personid': [1, 1],
            'date': ['2024-01-01'] * 2,
            'age': [25] * 2,
            'hgbType': ['HgbA', 'HgbS'],
            'value': ['>5', '<90'],
        })
        result = pivot_lab_data(df)
        assert result['HgbA'].iloc[0] == 5.0
        assert result['HgbS'].iloc[0] == 90.0


# ---------------------------------------------------------------------------
# run_lab_phenotyping (integration)
# ---------------------------------------------------------------------------

class TestRunLabPhenotyping:
    def test_long_format_pipeline(self):
        """Full pipeline from long-format data."""
        lab_data = pd.DataFrame({
            'personid': [1]*4 + [2]*4,
            'date': ['2024-01-01']*4 + ['2024-01-01']*4,
            'age': [25]*4 + [30]*4,
            'hgbType': ['HgbA', 'HgbS', 'HgbF', 'HgbC'] * 2,
            'value': [
                0, 85, 10, 0,     # Person 1: SCD_SCA
                55, 35, 5, 0,     # Person 2: S_Trait
            ],
        })
        person_df, row_df = run_lab_phenotyping(lab_data, input_format='long')
        assert len(person_df) == 2

        p1 = person_df[person_df['personid'] == 1]['LabPhenotype'].iloc[0]
        p2 = person_df[person_df['personid'] == 2]['LabPhenotype'].iloc[0]
        assert p1 == 'SCD_SCA'
        assert p2 == 'S_Trait'

    def test_wide_format_pipeline(self):
        """Full pipeline from wide-format data."""
        wide_data = pd.DataFrame({
            'personid': [1, 2],
            'date': ['2024-01-01', '2024-01-01'],
            'age': [25, 30],
            'HgbA': [0.0, 55.0],
            'HgbS': [85.0, 35.0],
            'HgbF': [10.0, 5.0],
            'HgbC': [0.0, 0.0],
        })
        person_df, row_df = run_lab_phenotyping(wide_data, input_format='wide')
        assert len(person_df) == 2
