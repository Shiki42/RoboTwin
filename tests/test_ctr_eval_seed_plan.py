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


def test_parallel_completion_order_cannot_change_selected_seeds():
    candidates=[1,2,3,4]
    results={3:dict(seed=3,status='accepted')}
    assert m.ordered_prefix(candidates,results,1)==[]
    results[1]=dict(seed=1,status='rejected')
    assert [r['seed'] for r in m.ordered_prefix(candidates,results,1)]==[1]
    results[2]=dict(seed=2,status='accepted')
    assert m.accepted_list(m.ordered_prefix(candidates,results,1),[],1)==[2]
    results[4]=dict(seed=4,status='accepted')
    assert m.accepted_list(m.ordered_prefix(candidates,results,1),[],1)==[2]
