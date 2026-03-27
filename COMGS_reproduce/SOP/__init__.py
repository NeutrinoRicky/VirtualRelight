def initialize_sop_phase1(*args, **kwargs):
    from .phase1_initializer import initialize_sop_phase1 as _initialize_sop_phase1

    return _initialize_sop_phase1(*args, **kwargs)


__all__ = ["initialize_sop_phase1"]
