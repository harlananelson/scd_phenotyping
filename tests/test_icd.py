"""
Tests for ICD-based phenotyping.

Uses synthetic data — no Git LFS dependencies.
"""

import pandas as pd
import numpy as np
import pytest

from scd_phenotyping.icd import run_icd_phenotyping, get_icd_regex
from scd_phenotyping.config import ICD_REGEX


# ---------------------------------------------------------------------------
# Synthetic test data generators
# ---------------------------------------------------------------------------

def make_icd_data():
    """
    Create synthetic ICD data with known phenotype distribution.

    Person 1: 10 SCD encounters → SCD
    Person 2: 3 TRAIT encounters → TRAIT
    Person 3: 5 THAL encounters → THAL
    Person 4: 2 SCD encounters (below threshold) → SCDX or UNK
    """
    rows = []

    # Person 1: SCD (D57.0 = Hb-SS with crisis)
    for i in range(10):
        rows.append({
            'personid': 1,
            'conditioncode_standard_id': 'D57.0',
            'effectivedate': f'2023-{(i % 12) + 1:02d}-15',
        })

    # Person 2: TRAIT (D57.3)
    for i in range(3):
        rows.append({
            'personid': 2,
            'conditioncode_standard_id': 'D57.3',
            'effectivedate': f'2023-{(i % 12) + 1:02d}-15',
        })

    # Person 3: THAL (D56.1)
    for i in range(5):
        rows.append({
            'personid': 3,
            'conditioncode_standard_id': 'D56.1',
            'effectivedate': f'2023-{(i % 12) + 1:02d}-15',
        })

    # Person 4: Only 2 SCD codes (below >2 threshold)
    for i in range(2):
        rows.append({
            'personid': 4,
            'conditioncode_standard_id': 'D57.1',
            'effectivedate': f'2023-{(i % 12) + 1:02d}-15',
        })

    return pd.DataFrame(rows)


def make_encounter_data():
    """Create synthetic encounter data matching the ICD data."""
    rows = []

    # Person 1: 15 OP encounters, 5 IP encounters
    for i in range(15):
        rows.append({
            'personid': 1,
            'classification_standard_primaryDisplay': 'Outpatient',
            'servicedate': f'2023-{(i % 12) + 1:02d}-15',
            'dischargedate': None,
        })
    for i in range(5):
        rows.append({
            'personid': 1,
            'classification_standard_primaryDisplay': 'Inpatient',
            'servicedate': f'2023-{(i % 5) + 1:02d}-01',
            'dischargedate': f'2023-{(i % 5) + 1:02d}-05',
        })

    # Person 2: 10 OP encounters
    for i in range(10):
        rows.append({
            'personid': 2,
            'classification_standard_primaryDisplay': 'Outpatient',
            'servicedate': f'2023-{(i % 12) + 1:02d}-15',
            'dischargedate': None,
        })

    # Person 3: 10 OP encounters
    for i in range(10):
        rows.append({
            'personid': 3,
            'classification_standard_primaryDisplay': 'Outpatient',
            'servicedate': f'2023-{(i % 12) + 1:02d}-15',
            'dischargedate': None,
        })

    # Person 4: 10 OP encounters
    for i in range(10):
        rows.append({
            'personid': 4,
            'classification_standard_primaryDisplay': 'Outpatient',
            'servicedate': f'2023-{(i % 12) + 1:02d}-15',
            'dischargedate': None,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestICDRegex:
    def test_scd_codes(self):
        """SCD regex matches D57.0, D57.1, D57.2, D57.4, D57.8, 282.41, 282.42, 282.6."""
        import re
        pattern = ICD_REGEX['SCD']
        assert re.search(pattern, 'D57.0')
        assert re.search(pattern, 'D57.1')
        assert re.search(pattern, 'D57.2')
        assert re.search(pattern, 'D57.4')
        assert re.search(pattern, 'D57.8')
        assert re.search(pattern, '282.41')
        assert re.search(pattern, '282.42')
        assert re.search(pattern, '282.6')
        # Should NOT match
        assert not re.search(pattern, 'D57.3')  # That's TRAIT
        assert not re.search(pattern, 'D56.1')  # That's THAL

    def test_trait_code(self):
        import re
        pattern = ICD_REGEX['TRAIT']
        assert re.search(pattern, 'D57.3')
        assert re.search(pattern, '282.5')
        assert not re.search(pattern, 'D57.0')

    def test_thal_code(self):
        import re
        pattern = ICD_REGEX['THAL']
        assert re.search(pattern, 'D56')
        assert re.search(pattern, 'D56.1')
        assert not re.search(pattern, 'D57.0')


class TestRunICDPhenotyping:
    def test_basic_pipeline(self):
        """Run full ICD pipeline with synthetic data."""
        icd_df = make_icd_data()
        enc_df = make_encounter_data()

        eval_df, enc_icd_df, enc_clean_df = run_icd_phenotyping(
            icd_df, enc_df, merge_encounters=False
        )

        assert len(eval_df) > 0
        assert 'IcdPheno' in eval_df.columns

        # Person 1 should be SCD (10 SCD encounters, >5% of total)
        p1 = eval_df[eval_df['personid'] == 1]
        assert len(p1) == 1
        assert p1['IcdPheno'].iloc[0] == 'SCD'

        # Person 2 should be TRAIT
        p2 = eval_df[eval_df['personid'] == 2]
        assert len(p2) == 1
        assert p2['IcdPheno'].iloc[0] == 'TRAIT'

        # Person 3 should be THAL
        p3 = eval_df[eval_df['personid'] == 3]
        assert len(p3) == 1
        assert p3['IcdPheno'].iloc[0] == 'THAL'

    def test_get_icd_regex(self):
        """get_icd_regex returns a copy of the patterns."""
        patterns = get_icd_regex()
        assert 'SCD' in patterns
        assert 'THAL' in patterns
        assert 'TRAIT' in patterns
        assert 'OTHER' in patterns
