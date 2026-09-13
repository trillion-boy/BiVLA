"""Effective time-per-step speedup: original pooled elapsed time per step divided by the configuration value. Includes recorded rollout overhead. Dots show measured speedup on a logarithmic axis; stems connect each measurement to the 1× original-policy reference. All 13 non-original configurations are shown in aligned backbone panels with shared axes. Color identifies the intervention family. Bars extend from zero for success and episode-step changes. Missing full-benchmark coverage is labeled explicitly. Backbone panels, left to right and then top to bottom: OpenVLA, SmolVLA, UniVLA. Figure and panel titles are omitted; panel identities are specified in this caption and the legends."""
from common import comparison

if __name__ == "__main__":
    comparison('step_speedup_libero', 'libero', 'speedup')
