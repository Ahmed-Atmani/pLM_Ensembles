from plm_models.ESM_plm import ESM1b_PLM, ESM1v_PLM, ESM2_PLM, ESMC_PLM
from plm_models.ProtT5_plm import ProtT5_PLM
from plm_models.ESMDance_plm import ESMDance_PLM, SeqDance_PLM
from plm_models.UniRep_plm import UniRep_PLM
from plm_models.CARP_plm import CARP_PLM
from plm_models.ProGen2_plm import ProGen2_PLM
from plm_models.Ankh3_plm import Ankh3_PLM
from plm_models.AntiBERTy_plm import AntiBERTy_PLM


# pLM name --> pLM class
PLMS_classes = {
    # General purpose
    "ProtT5": ProtT5_PLM,
    "ESM1b": ESM1b_PLM,
    "UniRep": UniRep_PLM,
    "CARP": CARP_PLM,
    "ProGen2": ProGen2_PLM,
    "Ankh3": Ankh3_PLM,

    # Immunology-specific
    "ESM1v": ESM1v_PLM,
    "AntiBERTy": AntiBERTy_PLM,

    # With added evolutionary information
    "ESM2": ESM2_PLM,
    "ESMC": ESMC_PLM,

    # With Structural information
    "ESMDance": ESMDance_PLM,
    "SeqDance": SeqDance_PLM,
}

PLMS_AVAILABLE = PLMS_classes.keys()

def load_plms(plm_names, small=False):
    """ Loads and returns pLMs

    Args:
        plm_names: list of plm names (e.g. ["ESM2", "ProtT5"])

    Returns:
        A dictionary mapping pLM names to their loaded pLM object
    """
    plms = {}
    for name in plm_names:
        plms[name] = PLMS_classes[name](small=small)
    return plms
