class CCH:
    def __init__(
        self,
        order: list[int],
        tail: list[int],
        head: list[int],
        filter_always_inf_arcs: bool,
    ) -> None: ...

class CCHMetric:
    def __init__(
        self,
        cch: CCH,
        weights: list[int],
    ) -> None:
        self.weights: list[int]

class CCHMetricPartialUpdater:
    def __init__(self, cch: CCH) -> None: ...
    def apply(self, metric: CCHMetric, updates: dict[int, int]) -> None: ...

class CCHQueryResult:
    distance: int | None
    node_path: list[int]
    arc_path: list[int]

class CCHQuery:
    def __init__(self, metric: CCHMetric) -> None: ...
    def run(self, source: int, target: int) -> CCHQueryResult: ...
    def run_multi_st_with_dist(
        self, sources: list[tuple[int, int]], targets: list[tuple[int, int]]
    ) -> CCHQueryResult:
        """run query with multiple sources and targets and distances."""

def compute_order_degree(
    node_count: int, tail: list[int], head: list[int]
) -> list[int]: ...
def compute_order_inertial(
    node_count: int,
    tail: list[int],
    head: list[int],
    latitude: list[float],
    longitude: list[float],
) -> list[int]: ...
