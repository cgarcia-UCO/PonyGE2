# Copied from https://stackoverflow.com/a/30134039

def partition(collection):
    if len(collection) == 1:
        yield [ collection ]
        return

    first = collection[0]
    for smaller in partition(collection[1:]):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[ first ] + subset]  + smaller[n+1:]
        # put `first` in its own subset
        yield [ [ first ] ] + smaller

def get_all_groups(collection):
    results = set()

    for p in partition(something):
        results.update(p) # I cannot do this, because lists are not hashable

    return results

if __name__ == '__main__':

    prueba = {1,2,3}
    prueba.add(4)
    # prueba.add([3,2,1]) # I cannot do this, because lists are not hashable

    something = list(range(1,5))

    for n, p in enumerate(partition(something), 1):
        print(n, sorted(p))

    # print(get_all_groups(something))