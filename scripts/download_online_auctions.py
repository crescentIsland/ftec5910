"""Download auction.csv locally; raw data is intentionally excluded from Git."""
from pathlib import Path
import hashlib,io,json,urllib.request,zipfile
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/"data/raw/online_auctions/auction.csv"
URL="https://www.kaggle.com/api/v1/datasets/download/onlineauctions/online-auctions-dataset"
EXPECTED="c05d10bafd4f33aadb2c70ab562a23937bd11c8d32a7c73244c829c06302486e"
if DEST.exists():
    digest=hashlib.sha256(DEST.read_bytes()).hexdigest()
    if digest!=EXPECTED:raise ValueError("Existing input differs from the analyzed snapshot; original preserved.")
    print("Verified existing auction.csv:",digest)
else:
    req=urllib.request.Request(URL,headers={"User-Agent":"Mozilla/5.0"})
    with urllib.request.urlopen(req,timeout=60) as response:payload=response.read(30_000_001)
    if len(payload)>30_000_000:raise ValueError("Archive exceeds 30 MB limit.")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        files=[entry for entry in archive.infolist() if Path(entry.filename).name=="auction.csv"]
        if len(files)!=1:raise ValueError("Expected exactly one auction.csv.")
        if files[0].file_size>20_000_000:raise ValueError("CSV exceeds 20 MB limit.")
        data=archive.read(files[0])
    digest=hashlib.sha256(data).hexdigest()
    if digest!=EXPECTED:raise ValueError("Provider snapshot changed; do not silently reproduce different data.")
    DEST.parent.mkdir(parents=True,exist_ok=True)
    with DEST.open("xb") as f:f.write(data)
    print("Downloaded and verified",DEST)
print("Provider lists license as Unknown; review source terms before redistribution.")
