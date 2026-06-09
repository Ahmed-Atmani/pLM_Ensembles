# This file was made in assistance with GenAI (Gemini)

from ensemble_models.representation_fusion_model import get_representation_fusion_factory
from ensemble_models.task_level_model import get_task_level_factory
from ensemble_models.meta_task_level_model import get_meta_task_factory
from ensemble_models.sequential_model import get_sequential_factory

from pipelines.representation_fusion_pipeline import RepresentationFusionEnsembleModelPipeline
from pipelines.task_level_pipeline import TaskLevelEnsembleModelPipeline
from pipelines.meta_task_level_pipeline import MetaTaskLevelEnsembleModelPipeline
from pipelines.sequential_pipeline import SequentialEnsembleModelPipeline

from pipelines.pipeline_super import ensemble_pipeline_kfold_runner
from plm_models.model_loader import PLMS_AVAILABLE


import itertools

# ==========================================
# 1. DEFINE PLM NAMES & THEMATIC GROUPS
# ==========================================
plm_catalog = [
    "ProtT5", "ESM1b", "UniRep", "CARP", "ProGen2", "Ankh3",
    "ESM1v", "AntiBERTy", "ESM2", "ESMC", "ESMDance", "SeqDance"
]

# Higher-order manual presets (Triplets or greater, plus special pairs)
manual_presets = [
    ["ESM2", "ProtT5", "ESM1v", "ESMDance", "AntiBERTy", "ProGen2"], # Best roundup
    ["ESM2", "ESMC", "ESM1b", "ESM1v", "ESMDance"],                  # Entire ESM family
    ["CARP", "ESMC", "ProGen2", "UniRep", "ProtT5"],                 # All architectures in one
    ["ESM1v", "AntiBERTy", "ESM2", "ProtT5"]                         # Antibody/Immunology-specific variants
]

# ==========================================
# 2. GENERATE COMBINATIONS
# ==========================================
# A. Single Model Baselines (n=12)
single_baselines = [[name] for name in plm_catalog]

# B. Pairwise Combinations (n=66)
pairwise_combinations = [list(pair) for pair in itertools.combinations(plm_catalog, 2)]

# Combine everything into a unified target suite
# (Using tuple-conversion to filter out duplicates in case a preset overlapped a pair)
seen = set()
all_combinations = []

for combo in single_baselines + pairwise_combinations + manual_presets:
    combo_tuple = tuple(sorted(combo))
    if combo_tuple not in seen:
        seen.add(combo_tuple)
        all_combinations.append(combo)

print(f"--> Total unique embedding combinations generated: {len(all_combinations)}")

# ==========================================
# 3. DEFINE ENSEMBLE CONFIGURATIONS
# ==========================================
# Register your meta-classifiers and their corresponding factories here.
# Add your 3rd and 4th ensemble methods to this list when ready.
ensemble_meta_methods = [
    {
        "name": "RF",
        "factory_fn": lambda plms: get_representation_fusion_factory(plm_names=plms),
        "pipeline_cls": RepresentationFusionEnsembleModelPipeline
    },
    {
        "name": "TL",
        "factory_fn": lambda plms: get_task_level_factory(plm_names=plms),
        "pipeline_cls": TaskLevelEnsembleModelPipeline
    },
    {
        "name": "ML",
        "factory_fn": lambda plms: get_meta_task_factory(plm_names=plms),
        "pipeline_cls": MetaTaskLevelEnsembleModelPipeline
    },
]

# ==========================================
# 4. EXECUTION PIPELINE MATRIX
# ==========================================
print("Starting execution pipeline matrix...")

for plm_combination in all_combinations:
    is_ensemble = len(plm_combination) > 1
    combo_id = "+".join(plm_combination)
    
    if not is_ensemble:
        # -------------------------------------------------------------
        # PASS A: Single Model Baseline Execution (No Fusion Needed)
        # -------------------------------------------------------------
        print(f"\n[BASELINE] Running individual pLM: {combo_id}")
        
        # NOTE: For individual models, you can default to your standard 
        # single-model pipeline, or pick one baseline factory (like TL or RF)
        # depending on how your baseline setup is structurally built.
        model_factory = get_task_level_factory(plm_names=plm_combination, mut_combination="difference")
        ensemble_pipeline_kfold_runner(
            model_factory=model_factory,
            pipeline_class=TaskLevelEnsembleModelPipeline,
            id=combo_id,
            model_name="Baseline"
        )
        
    else:
        # -------------------------------------------------------------
        # PASS B: Ensemble Combinations Matrix (Run across all 4 Meta-Classifiers)
        # -------------------------------------------------------------
        print(f"\n[ENSEMBLE] Processing combination group: {combo_id}")

        for meta in ensemble_meta_methods:
            print(f"  └─► Strategy: {meta['name']}")
            
            # Dynamically instantiate the correct factory using lambda configuration
            model_factory = meta["factory_fn"](plm_combination)
            
            ensemble_pipeline_kfold_runner(
                model_factory=model_factory,
                pipeline_class=meta["pipeline_cls"],
                id=combo_id,
                model_name=meta["name"]
            )

print("\nMatrix computation sequence complete!")