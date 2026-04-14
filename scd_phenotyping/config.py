"""
Configuration for Hemoglobinopathy Phenotyping
==============================================

Contains ICD regex patterns, encounter type mappings, and default column names.
All regex patterns have been validated - NO COMMAS in character classes.

Clinical Logic:
- ER encounters are EXCLUDED from phenotype counts (coding unreliable)
- IP, OB, OP encounters are INCLUDED in phenotype counts
- SCD classification requires >2 qualifying encounters AND >5% of (IP+OP) encounters
- SCDX classification requires >2 qualifying encounters but ≤5%
"""

from typing import Dict, List

# -----------------------------------------------------------------------------
# ICD REGEX PATTERNS (CORRECTED - no commas in character classes)
# -----------------------------------------------------------------------------
ICD_REGEX: Dict[str, str] = {
    # Sickle Cell Disease: D57.0, D57.1, D57.2, D57.4, D57.8 (ICD-10)
    #                      282.41, 282.42, 282.6 (ICD-9)
    'SCD':   r'D57\.[01248]|282\.4[12]|282\.6',
    
    # Thalassemia: D56.x (ICD-10)
    #              282.40, 282.43-282.49 (ICD-9)
    'THAL':  r'D56|282\.4[034579]',
    
    # Sickle Cell Trait: D57.3 (ICD-10), 282.5 (ICD-9)
    'TRAIT': r'D57\.3|282\.5',
    
    # Other Hemoglobinopathies: D58.x (ICD-10)
    #                           282.0-282.3, 282.7, 282.9 (ICD-9)
    'OTHER': r'D58|282\.[012379]'
}

# -----------------------------------------------------------------------------
# ENCOUNTER TYPE CONFIGURATION
# -----------------------------------------------------------------------------

# Default encounter type values (order: OP, IP, ER, OB)
# Users can override with their own values matching their data
DEFAULT_ENC_TYPE_VALUES: List[str] = [
    'Outpatient',
    'Inpatient',
    'Emergency',
    'Admitted for Observation'
]

# Encounter types that COUNT toward phenotype (ER excluded - coding unreliable)
# This is derived from enc_type_values at runtime
def get_phenotype_enc_types(enc_type_values: List[str]) -> List[str]:
    """Return encounter types that count toward phenotype (excludes ER)."""
    [OP, IP, ER, OB] = enc_type_values
    return [IP, OB, OP]  # Note: ER intentionally excluded

# -----------------------------------------------------------------------------
# DEFAULT COLUMN MAPPINGS
# -----------------------------------------------------------------------------
DEFAULT_COLUMN_MAP: Dict[str, str] = {
    'id': 'personid',
    'encType': 'classification_standard_primaryDisplay',
    'encStart': 'servicedate',
    'encEnd': 'dischargedate',
    'icdCode': 'conditioncode_standard_id',
    'icdTime': 'effectivedate'
}

