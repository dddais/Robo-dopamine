"""Partitions of accepted token-alignment planes; no grounding reinterpretation."""


def temporal_domains(mapping, scope):
    domain = set(mapping['target'][scope]) | set(mapping['negative'][scope])
    planes = []
    for record in mapping['alignment'][scope]:
        t, h, w = record['grid_thw']
        start, end = record['start'], record['end']
        if record['mosaic'] or t != 1 or h % 2 or w % 2 or end-start != (h//2)*(w//2):
            raise ValueError('Requires one non-mosaic merged token time plane per accepted record')
        ids = sorted(domain & set(range(start, end)))
        if ids:
            planes.append(ids)
    flattened = [i for plane in planes for i in plane]
    if not planes or len(flattened) != len(set(flattened)) or set(flattened) != domain:
        raise ValueError('Time planes must be a disjoint exact cover of the accepted visual domain')
    return planes


def partitioned_redistribute(weights, plane_masks, target, strength, redistribute):
    # Disjoint planes commute. The one-plane case calls the original operation
    # directly, preserving its floating-point arithmetic as well as its algebra.
    shifted = weights
    for plane in plane_masks:
        shifted = redistribute(shifted, plane, target, strength)
    return shifted
