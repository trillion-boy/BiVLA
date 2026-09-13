"""Total recorded episodes across all 14 configurations by model–environment setting. The hatched segment marks 100 missing episodes: OpenVLA LIBERO conservative fusion has only three of four suites. Figure and panel titles are omitted; panel identities are specified in this caption and the legends."""
from common import coverage

if __name__ == "__main__":
    coverage('evaluation_coverage')
