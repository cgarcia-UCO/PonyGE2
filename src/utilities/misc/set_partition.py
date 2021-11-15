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

    for p in partition(collection):
        # print(p)
        for p2 in p:
            # print('Insert', p2)
            results.update((tuple(p2),))

    return results

if __name__ == '__main__':

    for i in range(0, 10):
        something = list(range(1,i+2))

        # for n, p in enumerate(partition(something), 1):
        #     print(n, p)
        #
        # print('----------------------------------')
        groups = get_all_groups(something)
        print(i+1, len(groups), 2**(i+1)-1)