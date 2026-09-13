"""Counts of model–environment settings with lower, identical, or higher observed success than original. Each setting has equal weight. These are descriptive point-estimate comparisons, not significance tests. The incomplete configuration has 12 settings rather than 13. Figure and panel titles are omitted; panel identities are specified in this caption and the legends."""
from common import consistency

if __name__ == "__main__":
    consistency('cross_setting_consistency')
