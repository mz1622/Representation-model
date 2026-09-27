"""Display gradient scale/clipping without equating differently weighted losses."""
import argparse
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from foodcomp.research_r0 import digest,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    control=ROOT/'output/v9_r4/mlp60_views_weight0';candidate=ROOT/'output/v9_r4/mlp60_metabolome_weight05'
    old=ROOT/'output/v9_r2/mlp60_mae_width512'
    frames=[];manifests=[]
    for run in [control,candidate]:
        m=json.loads((run/'run_manifest.json').read_text())
        if m['status']!='complete' or m['test_opened']:raise ValueError('Completed closed-test runs required.')
        h=pd.read_csv(run/'history.csv');np.testing.assert_array_equal(h.epoch,np.arange(1,61))
        for column in ['mean_preclip_gradient_norm','gradient_clip_fraction']:
            if not np.isfinite(h[column]).all():raise ValueError('Invalid diagnostic history.')
        frames.append(h);manifests.append(m)
    # Zero-view arm is an exactly replayed parent trajectory, used only because it
    # logged clipping. Verify common histories again; stored-state audit is linked.
    original=pd.read_csv(old/'history.csv')
    for column in ['epoch','train_loss','validation_primary','validation_legacy_log_mae','learning_rate','training_tasks']:
        np.testing.assert_array_equal(original[column],frames[0][column])
    for key in ['data_hash','panel_hash','name_cache_hash','seed']:
        if manifests[0][key]!=manifests[1][key]:raise ValueError('Unmatched research inputs.')
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.2),layout='constrained')
    for frame,label,color in zip(frames,['Metabolome coefficient 1','Metabolome coefficient 0.5'],['#167d9a','#bd7736']):
        axes[0].plot(frame.epoch,frame.mean_preclip_gradient_norm,label=label,color=color,lw=1.8)
        axes[1].plot(frame.epoch,100*frame.gradient_clip_fraction,label=label,color=color,lw=1.8)
    axes[0].set(ylabel='Mean pre-clip gradient norm',yscale='log',title='Gradient scale (all model parameters)')
    axes[1].set(ylabel='Batches clipped at norm 1 (%)',ylim=(0,100),title='Clipping is part of the intervention')
    for ax in axes:ax.set_xlabel('Epoch');ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=9)
    fig.suptitle('Auxiliary weight: unchanged nutrition coefficient and 187-axis denominator',fontsize=12)
    args.output_dir.mkdir(parents=True)
    for suffix in ['png','svg']:fig.savefig(args.output_dir/f'diagnostics.{suffix}',dpi=170)
    plt.close(fig)
    write_json(args.output_dir/'manifest.json',{'status':'complete','control':str(control.relative_to(ROOT)),
        'candidate':str(candidate.relative_to(ROOT)),'control_parent_history_exact':True,
        'parent_model_state_audit':'reports/v9_r4_views_zero_replay_v1/verification.json',
        'history_sha256':[digest(run/'history.csv') for run in [control,candidate]],
        'script_sha256':digest(Path(__file__)),'complete_test_opened':False,
        'scope':'Matched trajectories. Weight0 views control exactly replayed original parent histories/saved model states. Different scalar training losses are not compared. Clipping frequency is descriptive, not proof of conflict or its removal.'})


if __name__=='__main__':main()
