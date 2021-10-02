# Clean Run, One Annotator
path_log=log/experiments/clean/one_anno
mkdir -p $path_log

# Clusterless Sampling
nohup python -u src/experiments/random_sampling_all_clustering.py clean_graphs 10 1 > $path_log/rs_fitted_run.log &
nohup python -u src/experiments/random_walk_all_clustering.py clean_graphs 10 1 > $path_log/rw_fitted_run.log &
nohup python -u src/experiments/page_rank_all_clustering.py clean_graphs 10 1 > $path_log/pr_fitted_run.log &
nohup python -u src/experiments/mrw_all_clustering.py clean_graphs 10 1 > $path_log/mrw_fitted_run.log &
nohup python -u src/experiments/merw_all_clustering.py clean_graphs 10 1 > $path_log/merw_fitted_run.log &

# DWUG Sampling
nohup python -u src/experiments/dwug_sampling_cc_nosplit_clustering.py clean_graphs 10 1 > $path_log/dwug_ccn_fitted_run.log &
nohup python -u src/experiments/dwug_sampling_cc_split_clustering.py clean_graphs 10 1 > $path_log/dwug_ccs_fitted_run.log &
nohup python -u src/experiments/dwug_sampling_ccc_clustering.py clean_graphs 10 1 > $path_log/dwug_ccc_fitted_run.log &
nohup python -u src/experiments/dwug_sampling_cw_clustering.py clean_graphs 10 1 > $path_log/dwug_cw_fitted_run.log &
nohup python -u src/experiments/dwug_sampling_lm_clustering.py clean_graphs 10 1 > $path_log/dwug_lm_fitted_run.log &
nohup python -u src/experiments/dwug_sampling_sbm_clustering.py clean_graphs 10 1 > $path_log/dwug_sbm_fitted_run.log &

# DWUG RS Sampling
nohup python -u src/experiments/dwug_rs_sampling_cc_nosplit_clustering.py clean_graphs 10 1 > $path_log/dwug_rs_ccn_fitted_run.log &
nohup python -u src/experiments/dwug_rs_sampling_cc_split_clustering.py clean_graphs 10 1 > $path_log/dwug_rs_ccs_fitted_run.log &
nohup python -u src/experiments/dwug_rs_sampling_ccc_clustering.py clean_graphs 10 1 > $path_log/dwug_rs_ccc_fitted_run.log &
nohup python -u src/experiments/dwug_rs_sampling_cw_clustering.py clean_graphs 10 1 > $path_log/dwug_rs_cw_fitted_run.log &
nohup python -u src/experiments/dwug_rs_sampling_lm_clustering.py clean_graphs 10 1 > $path_log/dwug_rs_lm_fitted_run.log &
nohup python -u src/experiments/dwug_rs_sampling_sbm_clustering.py clean_graphs 10 1 > $path_log/dwug_rs_sbm_fitted_run.log &

# Clean Run, Five Annotator
path_log=log/experiments/clean/five_anno
mkdir -p $path_log

# Clusterless Sampling
nohup python -u src/experiments/random_sampling_all_clustering.py clean_graphs 10 5 > $path_log/rs_fitted_run.log &
nohup python -u src/experiments/random_walk_all_clustering.py clean_graphs 10 5 > $path_log/rw_fitted_run.log &
nohup python -u src/experiments/page_rank_all_clustering.py clean_graphs 10 5 > $path_log/pr_fitted_run.log &
nohup python -u src/experiments/mrw_all_clustering.py clean_graphs 10 5 > $path_log/mrw_fitted_run.log &
nohup python -u src/experiments/merw_all_clustering.py clean_graphs 10 5 > $path_log/merw_fitted_run.log &

# DWUG Sampling
nohup python -u src/experiments/dwug_sampling_cc_nosplit_clustering.py clean_graphs 10 5 > $path_log/dwug_ccn_fitted_run.log &
nohup python -u src/experiments/dwug_sampling_cc_split_clustering.py clean_graphs 10 5 > $path_log/dwug_ccs_fitted_run.log &
nohup python -u src/experiments/dwug_sampling_ccc_clustering.py clean_graphs 10 5 > $path_log/dwug_ccc_fitted_run.log &
nohup python -u src/experiments/dwug_sampling_cw_clustering.py clean_graphs 10 5 > $path_log/dwug_cw_fitted_run.log &
nohup python -u src/experiments/dwug_sampling_lm_clustering.py clean_graphs 10 5 > $path_log/dwug_lm_fitted_run.log &
nohup python -u src/experiments/dwug_sampling_sbm_clustering.py clean_graphs 10 5 > $path_log/dwug_sbm_fitted_run.log &

# DWUG RS Sampling
nohup python -u src/experiments/dwug_rs_sampling_cc_nosplit_clustering.py clean_graphs 10 5 > $path_log/dwug_rs_ccn_fitted_run.log &
nohup python -u src/experiments/dwug_rs_sampling_cc_split_clustering.py clean_graphs 10 5 > $path_log/dwug_rs_ccs_fitted_run.log &
nohup python -u src/experiments/dwug_rs_sampling_ccc_clustering.py clean_graphs 10 5 > $path_log/dwug_rs_ccc_fitted_run.log &
nohup python -u src/experiments/dwug_rs_sampling_cw_clustering.py clean_graphs 10 5 > $path_log/dwug_rs_cw_fitted_run.log &
nohup python -u src/experiments/dwug_rs_sampling_lm_clustering.py clean_graphs 10 5 > $path_log/dwug_rs_lm_fitted_run.log &
nohup python -u src/experiments/dwug_rs_sampling_sbm_clustering.py clean_graphs 10 5 > $path_log/dwug_rs_sbm_fitted_run.log &

# Fitted Run, One Annotator 
path_log=log/experiments/fitted/one_anno 
mkdir -p $path_log

# Clusterless Sampling
nohup python -u src/experiments/random_sampling_all_clustering.py fitted_graphs 1 1 > $path_log/rs_fitted_run.log &
nohup python -u src/experiments/random_walk_all_clustering.py fitted_graphs 1 1 > $path_log/rw_fitted_run.log &
nohup python -u src/experiments/page_rank_all_clustering.py fitted_graphs 1 1 > $path_log/pr_fitted_run.log &
nohup python -u src/experiments/mrw_all_clustering.py fitted_graphs 1 1 > $path_log/mrw_fitted_run.log &
nohup python -u src/experiments/merw_all_clustering.py fitted_graphs 1 1 > $path_log/merw_fitted_run.log &

# DWUG Sampling
nohup python -u src/experiments/dwug_sampling_cc_nosplit_clustering.py fitted_graphs 1 1 > $path_log/dwug_ccn_fitted_run.log &
nohup python -u src/experiments/dwug_sampling_cc_split_clustering.py fitted_graphs 1 1 > $path_log/dwug_ccs_fitted_run.log &
nohup python -u src/experiments/dwug_sampling_ccc_clustering.py fitted_graphs 1 1 > $path_log/dwug_ccc_fitted_run.log &
nohup python -u src/experiments/dwug_sampling_cw_clustering.py fitted_graphs 1 1 > $path_log/dwug_cw_fitted_run.log &
nohup python -u src/experiments/dwug_sampling_lm_clustering.py fitted_graphs 1 1 > $path_log/dwug_lm_fitted_run.log &
nohup python -u src/experiments/dwug_sampling_sbm_clustering.py fitted_graphs 1 1 > $path_log/dwug_sbm_fitted_run.log &

# DWUG RS Sampling
nohup python -u src/experiments/dwug_rs_sampling_cc_nosplit_clustering.py fitted_graphs 1 1 > $path_log/dwug_rs_ccn_fitted_run.log &
nohup python -u src/experiments/dwug_rs_sampling_cc_split_clustering.py fitted_graphs 1 1 > $path_log/dwug_rs_ccs_fitted_run.log &
nohup python -u src/experiments/dwug_rs_sampling_ccc_clustering.py fitted_graphs 1 1 > $path_log/dwug_rs_ccc_fitted_run.log &
nohup python -u src/experiments/dwug_rs_sampling_cw_clustering.py fitted_graphs 1 1 > $path_log/dwug_rs_cw_fitted_run.log &
nohup python -u src/experiments/dwug_rs_sampling_lm_clustering.py fitted_graphs 1 1 > $path_log/dwug_rs_lm_fitted_run.log &
nohup python -u src/experiments/dwug_rs_sampling_sbm_clustering.py fitted_graphs 1 1 > $path_log/dwug_rs_sbm_fitted_run.log &

# Fitted Run, Five Annotator 
path_log=log/experiments/fitted/five_anno 
mkdir -p $path_log

# Clusterless Sampling
nohup python -u src/experiments/random_sampling_all_clustering.py fitted_graphs 1 5 > $path_log/rs_fitted_run.log &
nohup python -u src/experiments/random_walk_all_clustering.py fitted_graphs 1 5 > $path_log/rw_fitted_run.log &
nohup python -u src/experiments/page_rank_all_clustering.py fitted_graphs 1 5 > $path_log/pr_fitted_run.log &
nohup python -u src/experiments/mrw_all_clustering.py fitted_graphs 1 5 > $path_log/mrw_fitted_run.log &
nohup python -u src/experiments/merw_all_clustering.py fitted_graphs 1 5 > $path_log/merw_fitted_run.log &

# DWUG Sampling
nohup python -u src/experiments/dwug_sampling_cc_nosplit_clustering.py fitted_graphs 1 5 > $path_log/dwug_ccn_fitted_run.log &
nohup python -u src/experiments/dwug_sampling_cc_split_clustering.py fitted_graphs 1 5 > $path_log/dwug_ccs_fitted_run.log &
nohup python -u src/experiments/dwug_sampling_ccc_clustering.py fitted_graphs 1 5 > $path_log/dwug_ccc_fitted_run.log &
nohup python -u src/experiments/dwug_sampling_cw_clustering.py fitted_graphs 1 5 > $path_log/dwug_cw_fitted_run.log &
nohup python -u src/experiments/dwug_sampling_lm_clustering.py fitted_graphs 1 5 > $path_log/dwug_lm_fitted_run.log &
nohup python -u src/experiments/dwug_sampling_sbm_clustering.py fitted_graphs 1 5 > $path_log/dwug_sbm_fitted_run.log &

# DWUG RS Sampling
nohup python -u src/experiments/dwug_rs_sampling_cc_nosplit_clustering.py fitted_graphs 1 5 > $path_log/dwug_rs_ccn_fitted_run.log &
nohup python -u src/experiments/dwug_rs_sampling_cc_split_clustering.py fitted_graphs 1 5 > $path_log/dwug_rs_ccs_fitted_run.log &
nohup python -u src/experiments/dwug_rs_sampling_ccc_clustering.py fitted_graphs 1 5 > $path_log/dwug_rs_ccc_fitted_run.log &
nohup python -u src/experiments/dwug_rs_sampling_cw_clustering.py fitted_graphs 1 5 > $path_log/dwug_rs_cw_fitted_run.log &
nohup python -u src/experiments/dwug_rs_sampling_lm_clustering.py fitted_graphs 1 5 > $path_log/dwug_rs_lm_fitted_run.log &
nohup python -u src/experiments/dwug_rs_sampling_sbm_clustering.py fitted_graphs 1 5 > $path_log/dwug_rs_sbm_fitted_run.log &