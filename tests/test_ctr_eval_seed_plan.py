import importlib.util
from pathlib import Path
import pytest
p=Path(__file__).parents[1]/'scripts/ctr_eval_seed_plan.py';s=importlib.util.spec_from_file_location('plan',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_candidates_are_ascending_unique_disjoint_and_capped():
    train=list(range(100));a=m.candidates(train);assert a==list(range(100,600))
    train=[0,6,20,194];a=m.candidates(train);assert len(a)==len(set(a))==500 and not set(a)&set(train) and a==sorted(a)
def test_acceptance_never_uses_training_or_duplicate_seeds():
    assert m.accepted_list([dict(seed=1,status='rejected'),dict(seed=2,status='accepted')],[1])==[2]
    with pytest.raises(ValueError):m.accepted_list([dict(seed=1,status='accepted')],[1])
    with pytest.raises(ValueError):m.accepted_list([dict(seed=2,status='accepted')]*2,[1])
