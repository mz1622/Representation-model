"""Diagnose source signatures and flag near-name split candidates, never merge them."""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from foodcomp.research_r0 import ResearchData, VERSION, write_json
from foodcomp.research_text import prepare_names

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists(): raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    data = ResearchData(ROOT/'data/processed'/VERSION)
    text, cache = prepare_names(data, ROOT)
    labels = data.profiles.source_key.to_numpy()
    rows = []
    with threadpool_limits(limits=4):
        for name, features in [('name_only', text), ('observedness_only', data.observed.astype(np.float32))]:
            model = LogisticRegression(max_iter=500, class_weight='balanced', random_state=20260922)
            model.fit(features[data.train], labels[data.train])
            predicted = model.predict(features[data.validation])
            rows.append({'input': name, 'accuracy': accuracy_score(labels[data.validation], predicted),
                         'balanced_accuracy': balanced_accuracy_score(labels[data.validation], predicted),
                         'max_iterations_used': int(model.n_iter_.max())})
    write_json(args.output_dir/'source_probes.json', {'results': rows, 'balanced_chance': 1/len(np.unique(labels[data.train])),
        'interpretation': 'Predictability does not establish leakage: food geography, composition coverage and formatting are confounded.',
        'complete_test_opened': False})
    p = data.profiles
    train = p.iloc[data.train].drop_duplicates('original_name').reset_index(drop=True)
    valid = p.iloc[data.validation].drop_duplicates('original_name').reset_index(drop=True)
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), min_df=2, max_features=30000)
    xtrain = vectorizer.fit_transform(train.original_name.fillna(''))
    xvalid = vectorizer.transform(valid.original_name.fillna(''))
    nearest = NearestNeighbors(n_neighbors=1, metric='cosine', n_jobs=4).fit(xtrain)
    candidates = []
    for start in range(0,len(valid),256):
        distance,index = nearest.kneighbors(xvalid[start:start+256])
        for offset in np.flatnonzero(distance[:,0] <= .04):
            left,right=valid.iloc[start+offset],train.iloc[index[offset,0]]
            candidates.append({'validation_profile_id':left.profile_id,'validation_name':left.original_name,
                'train_profile_id':right.profile_id,'train_name':right.original_name,'cosine_similarity':1-float(distance[offset,0]),
                'decision':'candidate_only_requires_identity_review_no_merge'})
    pd.DataFrame(candidates,columns=['validation_profile_id','validation_name','train_profile_id','train_name','cosine_similarity','decision']).to_csv(args.output_dir/'near_name_candidates.csv',index=False)
    write_json(args.output_dir/'summary.json',{'near_name_candidates':len(candidates),'automatic_merges':0,'complete_test_opened':False})
    print(rows, 'near-name candidates:', len(candidates))

if __name__=='__main__':main()
