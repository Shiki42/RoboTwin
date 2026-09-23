"""CPU-only construction and validation of train-disjoint evaluation candidates."""
def candidates(training_seeds,limit=500):
    excluded=set(training_seeds)
    if len(excluded)!=len(training_seeds) or any(type(s) is not int or s<0 for s in excluded):
        raise ValueError('training seeds must be unique nonnegative integers')
    result=[];seed=0
    while len(result)<limit:
        if seed not in excluded:result.append(seed)
        seed+=1
    return result

def accepted_list(rows,training_seeds,target=100):
    selected=[r['seed'] for r in rows if r['status']=='accepted']
    if len(selected)!=len(set(selected)) or set(selected)&set(training_seeds):
        raise ValueError('duplicate or training-overlapping evaluation seed')
    if len(selected)>target:raise ValueError('too many evaluation seeds')
    return selected
