import importlib.util
import sys
import types
from pathlib import Path
import numpy as np
import pytest

root=Path(__file__).resolve().parents[1]
package=types.ModuleType('pair_test_envs'); package.__path__=[str(root/'envs')]
sys.modules['pair_test_envs']=package
spec=importlib.util.spec_from_file_location('pair_test_envs.paired_timing',root/'envs/paired_timing.py')
m=importlib.util.module_from_spec(spec); sys.modules[spec.name]=m;spec.loader.exec_module(m)


def program(n=20,k=30,tail=0,gate=False):
    l=np.zeros((n,16));r=np.zeros((k,16));t=np.zeros((tail,17))
    return m.Program(l,r,t,{'left':5,'right':5} if gate else {})


def execute(p,name='pick_dual_bottles',u=None):
    c=m.Clock(p,name,u)
    reasons=[]; commands=[]
    while not c.finished():
        controls,why,_=c.next();reasons.append(why);commands.append(controls)
        assert c.step<10000
    c.close()
    return c,np.array(reasons),commands


@pytest.mark.parametrize('u,expected',[(0,20),(1,-30),(None,0),(.5,-5)])
def test_offsets_and_hashes(u,expected):
    c,r,commands=execute(program(),u=u)
    starts={s:next(i for i,cmd in enumerate(commands) if s in cmd) for s in m.SIDES}
    assert starts['right']-starts['left']==expected
    assert not np.any(np.all(np.isin(r,[m.START,m.DONE]),axis=1))


def test_uniform_grid():
    values=[(i+50*k)/100 for i in range(50) for k in range(2)]
    assert sorted(values)==[i/100 for i in range(100)]


def test_barrier_and_tail_do_not_receive_idle_masks():
    c,r,cmd=execute(program(tail=10),'scan_object',u=0)
    barrier=next(e['step'] for e in c.events if e['event']=='scan_barrier_release')
    assert barrier==50
    assert np.all(r[20:50,0]==m.BARRIER)
    assert np.all(r[50:,1]==m.HOLD)
    masks,_=m.action_masks(r,list(range(0,len(r)+1,10)))
    assert not masks[2:,0].any()
    assert not masks[5:,1].any()


def test_workspace_mutex_wait_is_not_idle_and_releases_without_gap():
    c,r,cmd=execute(program(gate=True),'place_dual_shoes')
    assert np.any(r[:,1]==m.WORKSPACE)
    assert not np.any(np.all(r==m.WORKSPACE,axis=1))
    entry=[e for e in c.events if e['event']=='workspace_enter']
    exits=[e for e in c.events if e['event']=='workspace_exit']
    assert entry[1]['step']==exits[0]['step']
    mask,_=m.action_masks(r,list(range(len(r)+1)))
    assert not mask[r[:,1]==m.WORKSPACE,1].any()
    owner=None
    for e in c.events:
        if e['event']=='workspace_enter': assert owner is None;owner=e['arm']
        elif e['event']=='workspace_exit': assert owner==e['arm'];owner=None
    assert owner is None


def test_interval_boundary_keeps_effective_action():
    r=[[m.START,m.ACTIVE]]*9+[[m.ACTIVE,m.ACTIVE]]
    mask,overlap=m.action_masks(r,[0,10])
    assert not mask.any() and overlap.all()


def test_program_roundtrip(tmp_path):
    p=program(gate=True);path=tmp_path/'program.npz';p.save(path)
    assert m.Program.load(path).hashes()==p.hashes()
    assert m.Program.load(path).gate==p.gate
