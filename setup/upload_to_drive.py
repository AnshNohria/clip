import os
import mimetypes
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

SCOPES = ['https://www.googleapis.com/auth/drive']
print_lock = Lock()

# ─── CONFIG ──────────────────────────────────────────────
LOCAL_FOLDER     = 'D:/datasets'   # ← your folder
PARENT_FOLDER_ID = None            # ← Drive folder ID, or None for root
MAX_WORKERS      = 30              # ← parallel uploads (reduce to 5 if 429 errors)
# ─────────────────────────────────────────────────────────


def authenticate():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return creds


def build_service(creds):
    # Each thread needs its own service instance (not thread-safe to share)
    return build('drive', 'v3', credentials=creds)


def create_drive_folder(service, folder_name, parent_id=None):
    metadata = {
        'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder'
    }
    if parent_id:
        metadata['parents'] = [parent_id]
    folder = service.files().create(body=metadata, fields='id').execute()
    with print_lock:
        print(f"📁 Created folder: {folder_name}")
    return folder['id']


def upload_file_worker(args):
    """Worker: uploads one file. Each call builds its own Drive service."""
    creds, file_path, parent_id, pbar = args
    service = build_service(creds)

    file_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    mime_type, _ = mimetypes.guess_type(file_path)
    mime_type = mime_type or 'application/octet-stream'

    # 50 MB chunks for large files, 5 MB for small ones
    chunk_size = 50 * 1024 * 1024 if file_size > 10 * 1024 * 1024 else 5 * 1024 * 1024

    metadata = {'name': file_name, 'parents': [parent_id]}
    media = MediaFileUpload(file_path, mimetype=mime_type,
                            resumable=True, chunksize=chunk_size)
    try:
        service.files().create(body=metadata, media_body=media, fields='id').execute()
        with print_lock:
            pbar.update(file_size)
            pbar.set_postfix_str(f"✅ {file_name[:45]}")
    except Exception as e:
        with print_lock:
            pbar.write(f"❌ FAILED: {file_name} — {e}")


def collect_files(local_path, root_drive_id, creds):
    """
    Walk local_path, create matching Drive folder structure,
    and return list of (file_path, drive_parent_id) tuples.
    """
    service = build_service(creds)
    tasks = []
    
    # Cache mapping relative paths to their Drive folder IDs
    folder_cache = {'.': root_drive_id}

    for root, dirs, files in os.walk(local_path):
        rel = os.path.relpath(root, local_path)
        current_drive_id = folder_cache[rel]

        # Create Drive folders for all subdirectories of the current folder
        for d in dirs:
            dir_rel = os.path.normpath(os.path.join(rel, d))
            child_id = create_drive_folder(service, d, current_drive_id)
            folder_cache[dir_rel] = child_id

        for f in files:
            tasks.append((os.path.join(root, f), current_drive_id))

    return tasks


def upload_folder(local_path, parent_drive_id=None, max_workers=8):
    creds = authenticate()
    service = build_service(creds)

    folder_name = os.path.basename(os.path.abspath(local_path))
    root_drive_id = create_drive_folder(service, folder_name, parent_drive_id)

    print(f"\n🔍 Scanning {local_path} ...")
    tasks = collect_files(local_path, root_drive_id, creds)

    total_size = sum(os.path.getsize(fp) for fp, _ in tasks)
    print(f"📦 {len(tasks)} files | {total_size / (1024**3):.2f} GB total")
    print(f"🚀 Uploading with {max_workers} parallel workers...\n")

    with tqdm(total=total_size, unit='B', unit_scale=True,
              unit_divisor=1024, desc="Overall", dynamic_ncols=True) as pbar:

        work = [(creds, fp, fid, pbar) for fp, fid in tasks]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(upload_file_worker, w): w[1] for w in work}
            for future in as_completed(futures):
                future.result()

    print(f"\n🎉 Done! Root Drive folder ID: {root_drive_id}")
    return root_drive_id


if __name__ == '__main__':
    upload_folder(LOCAL_FOLDER, PARENT_FOLDER_ID, MAX_WORKERS)