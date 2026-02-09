# Username Configuration Summary

## ✅ Updated Configuration

All files have been updated with the correct usernames:

- **Hugging Face Username**: `UsamaMasood12ak` (with "ak" at the end)
- **GitHub Username**: `UsamaMasood12` (no "ak")

## Files Updated

### 1. `src/config.py` ✅
- **HF_DATASET_REPO**: `UsamaMasood12ak/engine-maintenance-dataset`
- **HF_MODEL_REPO**: `UsamaMasood12ak/engine-maintenance-model`
- **HF_SPACE_REPO**: `UsamaMasood12ak/engine-maintenance-space`

### 2. `README.md` ✅
- Updated all Hugging Face repo examples to use `UsamaMasood12ak`
- GitHub repository URLs use `UsamaMasood12`

### 3. `CONFIGURATION_GUIDE.md` ✅
- Updated examples to show `UsamaMasood12ak` for HF repos

### 4. `GITHUB_REPO_INFO.md` ✅
- Updated GitHub secrets examples to use `UsamaMasood12ak` for HF repos
- GitHub URLs use `UsamaMasood12`

## What You Need to Do

### 1. GitHub Secrets (in your GitHub repo settings)

When you add secrets to your GitHub repository, use these exact values:

- **`HF_TOKEN`**: Your Hugging Face access token
- **`HF_DATASET_REPO`**: `UsamaMasood12ak/engine-maintenance-dataset`
- **`HF_MODEL_REPO`**: `UsamaMasood12ak/engine-maintenance-model`
- **`HF_SPACE_REPO`**: `UsamaMasood12ak/engine-maintenance-space`

### 2. Local Environment Variables (optional)

If you want to override config.py, set these:

```bash
export HF_TOKEN="hf_your_token_here"
export HF_DATASET_REPO="UsamaMasood12ak/engine-maintenance-dataset"
export HF_MODEL_REPO="UsamaMasood12ak/engine-maintenance-model"
export HF_SPACE_REPO="UsamaMasood12ak/engine-maintenance-space"
```

### 3. GitHub Repository URLs

Your GitHub repository will be:
- **Repository**: `https://github.com/UsamaMasood12/engine-predictive-maintenance`
- **Actions**: `https://github.com/UsamaMasood12/engine-predictive-maintenance/actions`

### 4. Hugging Face Repos (will be created automatically)

After running the scripts, these will be created:
- **Dataset**: `https://huggingface.co/datasets/UsamaMasood12ak/engine-maintenance-dataset`
- **Model**: `https://huggingface.co/UsamaMasood12ak/engine-maintenance-model`
- **Space**: `https://huggingface.co/spaces/UsamaMasood12ak/engine-maintenance-space`

## Verification

All configuration files are now correctly set. You can proceed with:
1. Running the scripts locally
2. Pushing to GitHub
3. Setting up GitHub secrets
4. Running the GitHub Actions workflow
