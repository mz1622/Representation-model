"""Trace the largest nameKNN batch discrepancy without changing any scored artifacts."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from foodcomp.research_r0 import ResearchData,VERSION,digest,write_json
from foodcomp.research_alignment import NameSpace
from foodcomp.research_name_neighbors import ObservedAxisNeighbors
from sklearn.neighbors import NearestNeighbors


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True);run=ROOT/'output/v9_r5/retrieval_name_knn10'
    records=json.loads((run/'axis_fitting_manifest.json').read_text());axis=int(max(records,key=lambda x:x.get('candidate_vs_validation_max_raw_difference',0))['axis_index'])
    data=ResearchData(ROOT/'data/processed'/VERSION);ns=NameSpace(data,ROOT);text=ns.text
    train=data.train[data.observed[data.train,axis]];rows=data.jobs.loc[data.jobs.axis_index.eq(axis),'profile_index'].to_numpy()
    with threadpool_limits(limits=8):
        model=ObservedAxisNeighbors(text[train],data.values[train,axis],data.weights[train,axis],data.scale[axis])
        pv,iv,dv=model.predict(text[rows],return_neighbors=True);pc,ic,dc=model.predict(ns.features,return_neighbors=True)
        ids=ns.profile_name_ids[rows];delta=np.abs(pc[ids]-pv);j=int(delta.argmax());cid=int(ids[j]);query=text[rows[j]]
        np.testing.assert_array_equal(query,ns.features[cid])
        single=model.predict(query[None,:],return_neighbors=True)
        changed=np.array([not np.array_equal(np.sort(a),np.sort(b)) for a,b in zip(iv,ic[ids])])
        exact=((text[train].astype(np.float64)-query.astype(np.float64))**2).sum(1)
        stable=np.argsort(exact,kind='stable')[:10];cutoff=exact[stable[-1]]
        # Independent algorithm uses a per-query tree traversal, retaining fixed train order.
        tree=NearestNeighbors(n_neighbors=min(10,len(train)),algorithm='kd_tree',n_jobs=8).fit(text[train])
        tdv,tiv=tree.kneighbors(text[rows]);tdc,tic=tree.kneighbors(ns.features)
        tree_rows_equal=np.array([np.array_equal(a,b) for a,b in zip(tiv,tic[ids])]);tree_dist_equal=np.array([np.array_equal(a,b) for a,b in zip(tdv,tdc[ids])])
    private=ROOT/'data/local/research_diagnostics'/args.output_dir.name
    if private.exists():raise FileExistsError(private)
    private.mkdir(parents=True)
    np.savez(private/'case.npz',axis=np.array(axis),profile_index=np.array(rows[j]),training_rows=train,
        validation_neighbors=iv[j],candidate_neighbors=ic[cid],single_neighbors=single[1][0],validation_distances=dv[j],candidate_distances=dc[cid],
        direct_float64_squared_distances=exact,train_labels=data.values[train,axis],train_weights=data.weights[train,axis])
    changed_neighbors=np.array(sorted(set(iv[j])^set(ic[cid])),dtype=int)
    summary={'status':'complete','axis_index':axis,'canonical_axis':str(data.axes.canonical_name.iloc[axis]),
        'query_name_features_bitwise_equal':True,'axis_validation_rows':len(rows),'axis_prediction_mismatches':int(np.count_nonzero(delta)),
        'axis_neighbor_set_changes':int(changed.sum()),'max_raw_prediction_difference':float(delta[j]),
        'largest_case_neighbors_overlap':int(len(set(iv[j])&set(ic[cid]))),
        'largest_case_changed_neighbors_have_equal_direct_distances':bool(len(changed_neighbors) and np.all(exact[changed_neighbors]==cutoff)),
        'largest_case_changed_neighbor_count':len(changed_neighbors),
        'largest_case_tied_at_kth_distance':int((exact==cutoff).sum()),
        'largest_case_validation_neighbor_direct_distances':exact[iv[j]].tolist(),
        'largest_case_candidate_neighbor_direct_distances':exact[ic[cid]].tolist(),
        'single_query_matches_original_validation_indices':bool(np.array_equal(single[1][0],iv[j])),
        'single_query_matches_candidate_indices':bool(np.array_equal(single[1][0],ic[cid])),
        'tree_axis_validation_vs_candidate_indices_exact':bool(tree_rows_equal.all()),
        'tree_axis_validation_vs_candidate_distances_exact':bool(tree_dist_equal.all()),
        'private_case_sha256':digest(private/'case.npz'),'script_sha256':digest(Path(__file__)),
        'complete_test_opened':False,'scope':'Post-result largest-discrepancy axis/case diagnostic. No rescoring or model selection; one axis cannot prove all-axis batch invariance of an alternative backend.'}
    write_json(args.output_dir/'summary.json',summary);print(summary)


if __name__=='__main__':main()
