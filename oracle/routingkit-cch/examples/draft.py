import routingkit_cch as rk


def main():
    tail = [0, 1, 2]
    head = [1, 2, 3]
    weights = [10, 5, 7]
    node_count = 4

    order = rk.compute_order_degree(node_count, tail, head)
    cch = rk.CCH(order, tail, head, False)

    metric = rk.CCHMetric(cch, weights)
    updater = rk.CCHMetricPartialUpdater(cch)

    q = rk.CCHQuery(metric)
    res = q.run(0, 2)
    yield res
    del q, res

    updater.apply(metric, {0: 20, 2: 1})
    q = rk.CCHQuery(metric)
    res = q.run_multi_st_with_dist([(0, 0)], [(3, 0)])
    yield res
    del res

    res = q.run(3, 0)
    yield res


if __name__ == "__main__":
    for res in main():
        print("Here:")
        print(res.distance)
        print(res.node_path)
        print(res.arc_path)
        del res

    print("Done")
