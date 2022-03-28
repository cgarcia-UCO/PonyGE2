# Copied from https://miguendes.me/python-flatten-list

def rflatten(lst):
    """Flatten a list using recursion."""
    for item in lst:
        if isinstance(item, list):
            yield from rflatten(item)
        else:
            yield item
