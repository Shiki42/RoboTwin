"""User-confirmed shared50-seed Scan CTR recipe; CPU-only and deterministic."""

def recipe(runtime):
    def row(seed,variant,u=None,grid=-1):
        return dict(slot=seed,variant=variant,u=u,grid_index=grid)
    sequential=[row(i,'left_first',0.) for i in range(50)]+[row(i,'right_first',1.) for i in range(50)]
    concurrent=[row(i,'concurrent') for i in range(50)]
    ctr=[row(i,'ctr_'+str(k),(i+50*k)/100,i+50*k) for i in range(50) for k in range(2)]
    mixed=sequential[:34]+sequential[50:83]+concurrent[17:50]
    groups=dict(sequential=sequential,concurrent=concurrent,ctr=ctr,mixed=mixed)
    candidates=[]
    for i in range(50):
        variants=[dict(variant='left_first',first='left',delay_fraction=1.),dict(variant='right_first',first='right',delay_fraction=1.),dict(variant='concurrent')]
        variants += [dict(variant='ctr_'+str(k),u=(i+50*k)/100) for k in range(2)]
        candidates.append(dict(slot=i,seed=i,task='scan_object_ctr',variants=variants))
    return dict(schema='ctr.scan50.recipe.v1',runtime=runtime,slots=50,workers=1,candidates=candidates,
        prompt='Pick up the object with the left arm and the scanner with the right arm, align to scan, then return both items to their fixed positions and return home.',
        datasets={method:dict(repo_id=f'local/scan-object-ctr-{method}-{len(rows)}ep',episodes=rows) for method,rows in groups.items()})
