# This file was made in assistance with GenAI (Gemini)

from ensemble_models.representation_fusion_model import get_representation_fusion_factory
from pipelines.representation_fusion_pipeline import RepresentationFusionEnsembleModelPipeline
from pipelines.pipeline_super import ensemble_pipeline_kfold_runner

# ==========================================
# 1. DEFINE TARGET PLMS & MUTATION METHODS
# ==========================================
# Explicitly selected single pLMs for this run
target_plms = ["ESM2", "UniRep", "ESM1v", "Ankh3", "ESMC", "ProtT5", "ESM1b"]

# The three structural mutation combination strategies to evaluate
mut_strategies = ["append", "difference", "hybrid"]

# ==========================================
# 2. EXECUTION PIPELINE MATRIX
# ==========================================
print("Starting Representation Fusion (RF) Single-pLM Variant Matrix...")
print(f"Total experiments to execute: {len(target_plms) * len(mut_strategies)}")

for plm in target_plms:
    for strategy in mut_strategies:
        # Wrap in a single-element list as expected by the factory parameters
        plm_combination = [plm]
        
        # Unique identifier string for tracking logs and outputs
        # Format example: "esm2_append" or "ankh3_hybrid"
        experiment_id = f"{plm}_{strategy}"
        
        print(f"\n[RUNNING] RF Model | pLM: {plm} | Mutation Strategy: {strategy}")
        
        # Build the dynamic RF factory with the specific mutation combination rule
        model_factory = get_representation_fusion_factory(
            plm_names=plm_combination, 
            mut_combination=strategy
        )
        
        # Hand off execution loop to your k-fold runner
        ensemble_pipeline_kfold_runner(
            model_factory=model_factory,
            pipeline_class=RepresentationFusionEnsembleModelPipeline,
            id=experiment_id,
            model_name="RF"
        )

print("\nRF single-pLM mutation combination runner sequence complete!")