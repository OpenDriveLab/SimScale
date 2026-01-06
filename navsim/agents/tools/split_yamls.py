import os
import yaml

def quoted_presenter(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='"')

N = 32
root = f'{os.getenv("NAVSIM_DEVKIT_ROOT")}/navsim/planning/script/config/common/train_test_split'
tgt_yaml = 'navtrain'

# Load the original YAML file
with open(f'{root}/{tgt_yaml}.yaml', 'r') as file:
    data = yaml.safe_load(file)

# Generate and save sub-files
for i in range(N):
    data['defaults'][0]['scene_filter'] = f'{tgt_yaml}_sub{i + 1}'
    with open(f'{root}/{tgt_yaml}_sub{i + 1}.yaml', 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

yaml.add_representer(str, quoted_presenter)

# Load the original YAML file
with open(f'{root}/scene_filter/{tgt_yaml}.yaml', 'r') as file:
    data = yaml.safe_load(file)

tokens = data['tokens']

# Calculate chunk size and remainder
chunk_size = len(tokens) // N
remainder = len(tokens) % N

# Split tokens into N chunks with remainder appended to last chunk
split_tokens = []
start = 0
for i in range(N):
    # Last chunk gets remaining tokens
    end = start + chunk_size + (1 if i < remainder else 0)
    split_tokens.append(tokens[start:end])
    start = end

# Generate and save sub-files
for i, token_part in enumerate(split_tokens):
    data['tokens'] = token_part
    with open(f'{root}/scene_filter/{tgt_yaml}_sub{i + 1}.yaml', 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)