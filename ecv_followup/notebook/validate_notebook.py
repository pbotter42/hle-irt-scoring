"""Separate execution check; validation code is not included in the notebook."""
import hashlib, json, os, sys, tempfile, importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
import nbformat
from nbclient import NotebookClient

HERE=Path(__file__).resolve().parent
path=HERE/'hle_ecv_walkthrough.ipynb'
nb=nbformat.read(path,as_version=4)
nbformat.validate(nb)
assert not any('smoke test' in cell.source.lower() for cell in nb.cells)
with tempfile.TemporaryDirectory(prefix='hle-ecv-kernel-') as temp:
    spec=Path(temp)/'kernels/hle-ecv-validation'
    spec.mkdir(parents=True)
    (spec/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
                                                'display_name':'HLE ECV validation','language':'python'}))
    old=os.environ.get('JUPYTER_PATH','')
    os.environ['JUPYTER_PATH']=temp+(os.pathsep+old if old else '')
    try:
        NotebookClient(nb,timeout=300,kernel_name='hle-ecv-validation',
                       resources={'metadata':{'path':str(HERE)}}).execute()
    finally:
        if old:os.environ['JUPYTER_PATH']=old
        else:os.environ.pop('JUPYTER_PATH',None)
code=[c for c in nb.cells if c.cell_type=='code']
errors=[o for c in code for o in c.outputs if o.output_type=='error']
stderr=[o.text for c in code for o in c.outputs if o.output_type=='stream' and o.name=='stderr']
assert not errors and not stderr,(errors,stderr)
assert all(c.execution_count is not None for c in code)
figures=sum('image/png' in o.get('data',{}) for c in code for o in c.outputs)
assert figures==5
items=pd.read_csv(HERE/'results/item_ecv.csv')
ref=pd.read_csv(HERE/'data/item_ecv.csv')
assert len(items)==484 and items.item_id.tolist()==ref.item_id.tolist()
for col in ['general_slope','specific_slope','IECV_general','general_loading_standardized','specific_loading_standardized']:
    assert np.allclose(items[col],ref[col],atol=1e-10),col
assert items.IECV_general.between(0,1).all()
assert np.allclose(items.IECV_general,items.general_slope**2/(items.general_slope**2+items.specific_slope**2))
assert (items.general_slope<0).sum()==0 and (items.specific_slope<0).sum()==0
summary=pd.read_csv(HERE/'results/subtest_ecv.csv')
assert summary.iloc[:-1].Items.sum()==484
assert np.allclose(summary['General ECV']+summary['Specific ECV'],1)
refsum=pd.read_csv(HERE/'data/subtest_ecv.csv')
assert np.allclose(summary['General ECV'],refsum.ECV_general,atol=1e-10)
counts=summary.iloc[:-1].Items.to_numpy()
puc=1-(counts*(counts-1)).sum()/(484*483)
pair_group=items.category.to_numpy()
i,j=np.triu_indices(484,1)
assert np.isclose(puc,np.mean(pair_group[i]!=pair_group[j]))
nb.metadata.kernelspec={'name':'python3','language':'python','display_name':'Python 3'}
nbformat.write(nb,path)
report={'date':'2026-10-29','code_cells':len(code),'figures':figures,'all_cells_executed':True,
        'execution_errors':0,'warnings':stderr,'observed_answers':14760,'items':len(items),
        'overall_ECV':float(summary.iloc[-1]['General ECV']),'PUC':float(puc),
        'negative_specific_slopes_in_ECV':0,'excluded_items':29,'saved_ECV_values_reproduced':True,
        'scope':'Saved full-data state, fresh item E-step at q17, slope conversion, ECV, score comparisons, five figures. No refitting.',
        'notebook_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'fit_sha256':hashlib.sha256((HERE/'fit/selected_q11.npz').read_bytes()).hexdigest(),
        'versions':{k:importlib.metadata.version(k) for k in ['numpy','pandas','scipy','jax','matplotlib','nbformat','nbclient','ipykernel']}}
(HERE/'notebook_verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
