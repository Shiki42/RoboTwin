"""Exercise candidate orchestration without importing the GPU-only runtime."""
import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

def candidate_method():
    tree=ast.parse((Path(__file__).parents[1]/'envs/robot/planner.py').read_text())
    cls=next(n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name=='CuroboPlanner')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='plan_batch')
    module=ast.Module(body=[method],type_ignores=[]);ns={'np':np};exec(compile(module,'planner','exec'),ns);return ns['plan_batch']

def test_candidates_preserve_order_lengths_and_failures():
    calls=[]
    def plan(q,p,**kw):
        calls.append(p)
        return dict(status='Fail') if p==1 else dict(status='Success',position=np.ones((p+2,6))*p,velocity=np.zeros((p+2,6)))
    result=candidate_method()(SimpleNamespace(plan_path=plan),[0]*6,[0,1,2],arms_tag='right')
    assert calls==[0,1,2] and result['status']==['Success','Fail','Success']
    assert [len(p) for p in result['position']]==[2,0,4]

def test_infrastructure_errors_propagate():
    def plan(*args,**kwargs):raise RuntimeError('planner infrastructure failure')
    with pytest.raises(RuntimeError,match='infrastructure'):
        candidate_method()(SimpleNamespace(plan_path=plan),[0]*6,[0])
