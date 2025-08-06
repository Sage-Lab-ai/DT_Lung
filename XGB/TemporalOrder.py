from typing import Iterable, Sequence
import re


class TemporalOrder:

    def __init__(self, orders: Iterable[Sequence[str]]):
        self.orders = orders

    def __contains__(self, feature: str):
        return any([feature in order for order in self.orders])

    def __iter__(self):
        return iter(self.orders)

    def pre(self, feature: str) -> Sequence[str]:
        for order in self.orders:
            if feature in order:
                return order[:order.index(feature)]
        return []

    def post(self, feature: str) -> Sequence[str]:
        for order in self.orders:
            if feature in order:
                return order[order.index(feature)+1:]
        return []


def temporal_order_from_patterns(column_names: Iterable[str], patterns: Iterable[str]) -> TemporalOrder:
    """
    Constructs a TemporalOrder by grouping column names based on provided regex patterns.

    Example:
        patterns = ["H1_", "H2_", "H3_", "_y"]
        column_names = [
            "H1_A", "H2_A", "H3_A", "A_y",
            "H1_B", "H2_B", "H3_B", "B_y",
            "H1_C", "H2_C", "H3_C", "C_y",
        ]

        The temporal order will be:
        [
            ["H1_A", "H2_A", "H3_A", "A_y"],
            ["H1_B", "H2_B", "H3_B", "B_y"],
            ["H1_C", "H2_C", "H3_C", "C_y"],
        ]
    """
    compiled_patterns = [re.compile(pattern) for pattern in patterns]
    orders = {}
    for pattern in compiled_patterns:
        matched_cols = {pattern.sub('', col): col for col in column_names if pattern.search(col)}
        for base_feature, col in matched_cols.items():
            orders.setdefault(base_feature, []).append(col)

    return TemporalOrder(list(orders.values()))


if __name__ == "__main__":
    columns = [
        "H1_A", "H2_A", "H3_A", "A_y",
        "H1_B", "H2_B", "H3_B", "B_y",
        "H1_C", "H2_C", "H3_C", "C_y",
    ]

    order_pattern = ["H1_", "H2_", "H3_", "_y"]

    temporal_order = temporal_order_from_patterns(columns, order_pattern)

    print(temporal_order.orders)