"""Guarded reuse configuration sensitivity across all model–environment settings. Left: success change in percentage points; right: effective time-per-step speedup. Lines connect measured configurations only. Categorical reuse/fusion settings are not a continuous numerical scale. Figure and panel titles are omitted; panel identities are specified in this caption and the legends."""
from common import ablation

if __name__ == "__main__":
    ablation('ablation_guarded_reuse', 4)
