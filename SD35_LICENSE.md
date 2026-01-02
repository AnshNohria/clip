# IMPORTANT: Accept SD 3.5 License First!

## Error You're Seeing:
```
401 Client Error: Unauthorized
Cannot access gated repo
Access to model stabilityai/stable-diffusion-3.5-medium is restricted
```

## Solution:

### Step 1: Accept the License (ONE TIME ONLY)
1. **Go to:** https://huggingface.co/stabilityai/stable-diffusion-3.5-medium
2. **Log in** with your HuggingFace account (same account as your HF_TOKEN)
3. **Click** the blue "Agree and access repository" button
4. **Wait** 2-3 minutes for access to be granted

### Step 2: Verify Your Token
Make sure your `.env` file has the correct token:
```bash
cat .env | grep HF_TOKEN
```

The token should match the HuggingFace account you used in Step 1!

### Step 3: Run Download Script
After accepting the license:

```bash
source /home/jovyan/clip/clip-venv/bin/activate
python download_sd35.py
```

Or download all models:
```bash
python download_all_models.py
```

---

## Why This Happens
Stable Diffusion 3.5 Medium is a "gated" model - Stability AI requires users to:
- Have a HuggingFace account
- Accept their license terms
- Be authenticated with a token

This is a one-time process per HuggingFace account.

---

## Still Not Working?

1. **Check your token is valid:**
   ```bash
   python -c "from huggingface_hub import whoami; import os; from dotenv import load_dotenv; load_dotenv(); print(whoami(os.getenv('HF_TOKEN')))"
   ```

2. **Make sure you accepted the license** on the website (check for confirmation email)

3. **Try logging in manually:**
   ```bash
   huggingface-cli login
   ```
   Paste your token when prompted.

4. **Restart and try again** after 5-10 minutes (access approval can be delayed)
