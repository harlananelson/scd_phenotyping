"""
SCD Phenotyping Package
=======================

Source-of-truth implementation for phenotyping patients based on ICD codes
and hemoglobin electrophoresis lab results for hemoglobinopathies including
Sickle Cell Disease (SCD), Thalassemia, Sickle Cell Trait, and other
hemoglobinopathies.

Version: 2.0.0

ICD pipeline: DuckDB-based encounter-linked phenotyping (icd.py)
Lab pipeline: Ratio-based, transfusion-aware classification (lab.py)
Combined: ICD + Lab consensus phenotyping (combined.py)
"""

from .icd import run_icd_phenotyping, get_icd_regex
from .lab import (
    classify_lab_row,
    aggregate_lab_phenotypes,
    calculate_ratios,
    run_lab_phenotyping,
    pivot_lab_data,
    get_best_hgb_row,
    infer_transfusion_status,
    add_actual_transfusion,
    add_hu_proximity,
    extract_numeric,
)
from .combined import run_combined_phenotyping, SCD_LAB_PHENOTYPES
from .utils import clean_encounters, clean_merge_encounters, get_con
from .config import ICD_REGEX, DEFAULT_ENC_TYPE_VALUES, DEFAULT_COLUMN_MAP

__version__ = "2.0.0"

__all__ = [
    # ICD phenotyping
    'run_icd_phenotyping',
    'get_icd_regex',
    # Lab phenotyping
    'classify_lab_row',
    'aggregate_lab_phenotypes',
    'calculate_ratios',
    'run_lab_phenotyping',
    'pivot_lab_data',
    'get_best_hgb_row',
    'infer_transfusion_status',
    'add_actual_transfusion',
    'add_hu_proximity',
    'extract_numeric',
    # Combined phenotyping
    'run_combined_phenotyping',
    'SCD_LAB_PHENOTYPES',
    # Utilities
    'clean_encounters',
    'clean_merge_encounters',
    'get_con',
    # Configuration
    'ICD_REGEX',
    'DEFAULT_ENC_TYPE_VALUES',
    'DEFAULT_COLUMN_MAP',
]

