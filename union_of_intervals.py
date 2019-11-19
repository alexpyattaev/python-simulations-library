def union_of_intervals(data):
    """Find the union of intervals in the given data sequence
    
    :return iterator over unified intervals"""
    it = iter(sorted(data))
    a, b = next(it)
    for c, d in it:
        if b >= c:  # Use `if b > c` if you want (1,2), (2,3) not to be
                    # treated as intersection.
            b = max(b, d)
        else:
            yield a, b
            a, b = c, d
    yield a, b


def unify_intervals(data):
    """Unify all intervals in given data sequence. Intervals 
    should be overlapping or continuous

    :return the unified sequence start and end points 
    :raises ValueError if can not unify"""
    it = iter(sorted(data))
    a, b = next(it)
    for c, d in it:
        if b >= c:  # Use `if b > c` if you want (1,2), (2,3) not to be
                    # treated as intersection.
            b = max(b, d)
        else:
            raise ValueError(b)
    return a, b



if __name__ == "__main__":
    data = [
        [10,100],
        [90, 160],
        [50, 60],
        [170, 200]
    ]
    print(list(union_of_intervals(data)))

    print(unify_intervals(data[0:3]))
    print(unify_intervals(data))