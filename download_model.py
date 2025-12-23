from huggingface_hub import snapshot_download

# Read token from mounted secret
with open('/run/secrets/hf_token', 'r') as f:
    token = f.read().strip()

# Download model to local path
snapshot_download(
    'facebook/sam-audio-large',
    local_dir='/src/weights/sam-audio-large',
    token=token
)

print("Model downloaded successfully!")