import math
from collections import Counter


def profile(rows, columns, weight_field=None):
    result = {}
    for column in columns:
        counts = Counter()
        missing = 0.0
        total = 0.0
        for row in rows:
            weight = row.get(weight_field, 1) if weight_field else 1
            if not math.isfinite(weight) or weight < 0:
                raise ValueError('人群权重无效')
            total += weight
            if row.get(column) is None:
                missing += weight
            else:
                counts[str(row[column])] += weight
        result[column] = {'counts': dict(sorted(counts.items())), 'missing_weight': missing, 'total_weight': total}
    return result


def total_variation(left, right):
    total_left, total_right = sum(left.values()), sum(right.values())
    if not total_left or not total_right:
        return None
    return sum(abs(left.get(key, 0) / total_left - right.get(key, 0) / total_right)
               for key in set(left) | set(right)) / 2


def inspect(document):
    source = profile(document['source'], document['columns'])
    target = profile(document['target'], document['columns'], 'population_weight')
    return {'source': source, 'target': target, 'marginal_distance':
            {column: total_variation(source[column]['counts'], target[column]['counts']) for column in document['columns']}}
