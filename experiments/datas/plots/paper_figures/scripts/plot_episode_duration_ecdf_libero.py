"""Empirical CDF of all recorded episode durations, including failures, normalized by the original mean within each model and environment. Six prespecified representatives: O, F20, A2, D1, RM, FT (see code key). Duration reflects termination and failure as well as computation; read alongside success. Backbone panels, left to right and then top to bottom: OpenVLA, SmolVLA, UniVLA. Figure and panel titles are omitted; panel identities are specified in this caption and the legends."""
from common import duration_ecdf

if __name__ == "__main__":
    duration_ecdf('episode_duration_ecdf_libero', 'libero')
