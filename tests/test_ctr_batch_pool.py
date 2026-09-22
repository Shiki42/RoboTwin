import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

@pytest.mark.parametrize('accepted_seeds', [{0,2},set()])
def test_finite_pool_keeps_slot_on_rejection_and_never_repeats_seed(tmp_path,monkeypatch,accepted_seeds):
    scripts=Path(__file__).parents[1]/'script';monkeypatch.syspath_prepend(str(scripts))
    spec=importlib.util.spec_from_file_location('batch',scripts/'collect_ctr_batch.py')
    batch=importlib.util.module_from_spec(spec);spec.loader.exec_module(batch)
    from scan50_recipe import recipe
    plan=recipe('/python');plan.update(slots=2,slot_jobs=plan['slot_jobs'][:2],candidate_seeds=[0,1,2])
    plan_path=tmp_path/'plan.json';plan_path.write_text(json.dumps(plan));output=tmp_path/'collection';calls=[]
    def execute(command,**kwargs):
        i=int(command[command.index('--seed-index')+1]);slot=int(command[command.index('--slot')+1]);seed=plan['candidate_seeds'][i]
        calls.append((seed,slot));path=Path(command[command.index('--output')+1]);path.mkdir()
        result=dict(seed=seed,slot=slot,path=str(path),status='accepted' if seed in accepted_seeds else 'rejected',variants=[{}]*5)
        (path/'candidate.json').write_text(json.dumps(result));return SimpleNamespace(returncode=0)
    monkeypatch.setattr(batch.subprocess,'run',execute)
    monkeypatch.setattr(batch.sys,'argv',['batch','--plan',str(plan_path),'--output',str(output)])
    if accepted_seeds:
        batch.main();assert calls==[(0,0),(1,1),(2,1)]
        result=json.loads((output/'complete.json').read_text());assert result['selected_seeds']==[0,2] and result['episodes']==10
    else:
        with pytest.raises(RuntimeError,match='pool exhausted'):batch.main()
        assert calls==[(0,0),(1,0),(2,0)] and not (output/'complete.json').exists()
    assert len(json.loads((output/'attempts.json').read_text()))==3
