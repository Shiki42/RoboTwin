import numpy as np
import pytest
from envs.ctr_timing import StagedProgram,StageClock,delay_masks
from envs.paired_timing import START,DONE,BARRIER,HOLD,ACTIVE

def program():
    arrays={key:np.zeros((n,16),dtype=np.float64) for key,n in (
        ('prepare/left',10),('prepare/right',20),('scan_align/left',0),('scan_align/right',3),
        ('put_back/left',6),('put_back/right',4))}
    return StagedProgram([dict(name='prepare',independent=True),dict(name='scan_align',independent=False),
                          dict(name='put_back',independent=True)],arrays)

@pytest.mark.parametrize('u,first,fraction',[(0,'left',1),(1,'right',1),(1/3,'left',0),(.2,'left',.4),(.8,'right',.7)])
def test_schedule_preserves_controls_and_reuses_fraction(u,first,fraction):
    p=program();clock=StageClock(p,u=u)
    assert clock.first==first and clock.fraction==pytest.approx(fraction)
    steps=[]
    while not clock.finished():
        stage=clock.name
        controls,why,indices,end=clock.next()
        steps.append((stage,controls,why))
        if end:clock.advance_stage()
    assert clock.close()==p.hashes()
    assert clock.schedules[2]['raw_delay_steps']==pytest.approx(fraction*(6 if first=='left' else 4))
    assert clock.schedules[1]['start_step']==max(clock.schedules[0]['durations'][s]+(clock.schedules[0]['delay_steps'] if s!=first else 0) for s in ('left','right'))
    if u in (0,1):
        assert all(len(controls)<=1 for _,controls,_ in steps)

def test_idle_only_imposed_delay_and_entire_action_interval():
    why=[[START,ACTIVE],[START,ACTIVE],[DONE,BARRIER],[HOLD,START],[ACTIVE,START]]
    assert delay_masks(why,[0,2,3,5]).tolist()==[[True,False],[False,False],[False,True]]
    assert delay_masks(why,[0,3,5]).tolist()==[[False,False],[False,True]]

def test_barrier_cannot_be_released_early():
    with pytest.raises(RuntimeError,match='barrier'):
        StageClock(program()).advance_stage()

@pytest.mark.parametrize('arguments', [{'u':-0.1},{'u':1.1},{'u':float('nan')},
    {'delay_fraction':.5},{'first':'left','delay_fraction':1.1},
    {'first':'right','delay_fraction':.2,'u':.5}])
def test_ambiguous_and_out_of_range_timing_is_rejected(arguments):
    with pytest.raises(ValueError):
        StageClock(program(),**arguments)
