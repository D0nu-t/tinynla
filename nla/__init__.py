# """
# nla — Natural Language Autoencoder package.

# Import from here rather than submodules to insulate calling code from
# internal refactors.
# """

# # Extraction
# from nla.activations import (
#     ActivationExtractor,
#     MultiLayerExtractor,
# )

# # Hooks
# from nla.hooks import (
#     ActivationHook,
#     MultiLayerHook,
# )

# # Reconstruction
# from nla.reconstructor import (
#     TokenLevelReconstructor,    # v3 primary — distilgpt2 + TransformerDecoder
#     ActivationReconstructor,    # legacy pooled — DistilBERT + MLP
# )

# # Patching
# from nla.patching import (
#     ActivationPatcher,              # canonical alias → SequenceInterpolationPatcher
#     SequenceInterpolationPatcher,   # v3 primary — per-position interpolation
#     InterpolationPatcher,           # legacy pooled broadcast interpolation
#     BroadcastPatcher,               # legacy full-replacement broadcast
#     LastTokenPatcher,               # legacy last-token only
# )

# # Losses
# from nla.losses import (
#     masked_sequence_cosine_loss,    # v3 primary — variable-length with mask
#     sequence_cosine_loss,           # uniform-length sequence cosine
#     cosine_loss,                    # pooled cosine
#     combined_loss,                  # α·cosine + (1-α)·MSE
# )

# # Metrics
# from nla.metrics import (
#     cosine_similarity_metric,
#     aggregate_metrics,
# )

# # Evaluation — sequence mode (v3)
# from nla.evaluation import (
#     evaluate_condition_sequence,
#     evaluate_all_conditions_sequence,
#     run_interpolation_sweep_sequence,
#     # shared scalar metrics
#     kl_divergence,
#     topk_overlap,
#     logit_cosine_similarity,
#     perplexity_shift,
#     # legacy pooled evaluation
#     evaluate_condition,
#     evaluate_all_conditions,
#     run_interpolation_sweep,
# )

# # Dataset
# from nla.dataset import (
#     SequenceActivationDataset,  # v3 primary
#     ActivationDataset,          # legacy pooled
#     sequence_collate,
#     pooled_collate,
#     save_dataset,
# )

# # Labeling
# from nla.labeler import SemanticLabeler

# # Tracking
# from nla.tracking import WandbTracker

# # Utilities
# from nla.utils import (
#     load_config,
#     resolve_device,
#     set_seed,
# )

# __all__ = [
#     # Extraction
#     "ActivationExtractor", "MultiLayerExtractor",
#     # Hooks
#     "ActivationHook", "MultiLayerHook",
#     # Reconstruction
#     "TokenLevelReconstructor", "ActivationReconstructor",
#     # Patching
#     "ActivationPatcher", "SequenceInterpolationPatcher",
#     "InterpolationPatcher", "BroadcastPatcher", "LastTokenPatcher",
#     # Losses
#     "masked_sequence_cosine_loss", "sequence_cosine_loss",
#     "cosine_loss", "combined_loss",
#     # Metrics
#     "cosine_similarity_metric", "aggregate_metrics",
#     # Evaluation
#     "evaluate_condition_sequence", "evaluate_all_conditions_sequence",
#     "run_interpolation_sweep_sequence",
#     "kl_divergence", "topk_overlap", "logit_cosine_similarity", "perplexity_shift",
#     "evaluate_condition", "evaluate_all_conditions", "run_interpolation_sweep",
#     # Dataset
#     "SequenceActivationDataset", "ActivationDataset",
#     "sequence_collate", "pooled_collate", "save_dataset",
#     # Labeling
#     "SemanticLabeler",
#     # Tracking
#     "WandbTracker",
#     # Utils
#     "load_config", "resolve_device", "set_seed",
# ]