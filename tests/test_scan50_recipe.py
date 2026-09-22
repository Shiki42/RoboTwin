import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('recipe',Path(__file__).parents[1]/'script/scan50_recipe.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def test_requested_counts_seed_pairing_order_and_uniform_grid():
    p=module.recipe('/python');d=p['datasets']
    assert {k:len(v['episodes']) for k,v in d.items()}==dict(sequential=100,concurrent=50,ctr=100,mixed=100)
    s=d['sequential']['episodes'];assert [r['slot'] for r in s[:50]]==list(range(50))==[r['slot'] for r in s[50:]]
    assert all(r['variant']=='left_first' for r in s[:50]) and all(r['variant']=='right_first' for r in s[50:])
    c=d['ctr']['episodes'];assert sorted(r['grid_index'] for r in c)==list(range(100))
    for i in range(50):assert [(r['slot'],r['u']) for r in c[2*i:2*i+2]]==[(i,i/100),(i,(i+50)/100)]
    m=d['mixed']['episodes'];assert len({r['slot'] for r in m})==50
    assert m==s[:34]+s[50:83]+d['concurrent']['episodes'][17:50]
    assert len({(r['slot'],r['variant']) for r in m})==100
    assert len(p['slot_jobs'])==50 and all(len(c['variants'])==5 for c in p['slot_jobs'])
    assert p['candidate_seeds']==list(range(200))
    assert module.candidate_job(p,0,1)['seed']==1
    assert module.candidate_job(p,0,1)['variants']==module.candidate_job(p,0,0)['variants']
    assert module.candidate_job(p,1,7)['variants'][-2]['u']==.01
