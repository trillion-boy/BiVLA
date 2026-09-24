Tables I and II of the paper show, for each trick, only the setting with the highest success. The tables below list all 13 settings and the original for every backbone and environment.

- **Success %**: episodes solved by the benchmark's own criterion. For a trick setting, the parentheses give the change against the original on the same episodes and the two-sided exact McNemar p-value on the episodes that flipped. Green marks a significant gain and red a significant loss (p < 0.05). A star marks the setting shown in the paper's table (best of its trick, ties broken by fewer steps).
- **ms / step**: wall-clock latency per environment step, the episode time divided by its steps. As in the paper, latency is comparable only within one backbone and environment, and a few cells are not read (see the implementation details).
- **Avg. steps**: mean episode length. Failed episodes run to the step cap, so this column follows success.
