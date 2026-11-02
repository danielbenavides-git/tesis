"""One-off: copy "Trainable parameters" from each run's tables/hyperparameters.txt (read from git HEAD, so it works
after `git rm`) into that run's config.json as n_params. Run from the repo root before the commit that deletes the
.txt files."""
import json
import re
import subprocess
from pathlib import Path

files = subprocess.run(["git", "ls-tree", "-r", "--name-only", "HEAD", "experiments"], capture_output=True,
                       text=True, check=True).stdout.split()
done = 0
for f in (f for f in files if f.endswith("tables/hyperparameters.txt")):
    text = subprocess.run(["git", "show", f"HEAD:{f}"], capture_output=True, text=True, check=True).stdout
    match = re.search(r"Trainable parameters\s*\|\s*([\d,]+)", text)
    config_path = Path(f).parent.parent / "config.json"
    if not match or not config_path.exists():
        print(f"skip {f}")
        continue
    raw = config_path.read_bytes().decode()
    config = json.loads(raw)
    config["n_params"] = int(match.group(1).replace(",", ""))
    eol = "\r\n" if "\r\n" in raw else "\n"
    config_path.write_bytes(json.dumps(config, indent=2, default=str).replace("\n", eol).encode())
    done += 1
print(f"n_params added to {done} config.json files")
