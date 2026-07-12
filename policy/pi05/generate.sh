data_dir=${1}
repo_id=${2}
env -u LD_LIBRARY_PATH uv run examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py --raw_dir $data_dir --repo_id $repo_id
