"""Success point estimates for every configuration in Spatial, Object, Goal, and Long (libero_10). Each available suite has 100 episodes. A missing point means the suite was not recorded. No across-seed uncertainty is available. Figure and panel titles are omitted; panel identities are specified in this caption and the legends."""
from common import suite_success

if __name__ == "__main__":
    suite_success('libero_suites_univla', 'univla_libero')
