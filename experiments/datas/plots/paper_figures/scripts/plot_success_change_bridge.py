"""Success-rate change relative to each original policy, in percentage points. All 13 non-original configurations are shown in aligned backbone panels with shared axes. Color identifies the intervention family. Bars extend from zero for success and episode-step changes. Missing full-benchmark coverage is labeled explicitly. Backbone panels, left to right and then top to bottom: CogACT, CronusVLA, MiniVLA, OpenVLA, SpatialVLA, UniVLA. Figure and panel titles are omitted; panel identities are specified in this caption and the legends."""
from common import comparison

if __name__ == "__main__":
    comparison('success_change_bridge', 'bridge', 'delta')
