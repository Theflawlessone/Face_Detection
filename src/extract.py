from pathlib import Path
import tarfile

from src.paths import RAW_DIR, BASE_DIR


def safe_extract(tar_path: Path, extract_to: Path):
    """
    Safely extract a tar archive into `extract_to`
    """
    with tarfile.open(tar_path, "r:*") as tar:
        for member in tar.getmembers():
            member_path = (extract_to / member.name).resolve()
            if not str(member_path).startswith(str(extract_to.resolve())):
                raise Exception(f"Unsafe path in {tar_path}: {member.name}")
        tar.extractall(extract_to)
    print(f"Extracted {tar_path.name}")


def main():
    print("BASE_DIR:", BASE_DIR)
    print("RAW_DIR:", RAW_DIR)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    archives = [
        RAW_DIR / "part1.tar.gz",
        RAW_DIR / "part2.tar.gz",
        RAW_DIR / "part3.tar.gz",
    ]

    # check for missing archives
    missing = [arc for arc in archives if not arc.exists()]
    if missing:
        print("Missing dataset archives:")
        for m in missing:
            print(f"   - {m}")
        print("\nPlace all UTKFace .tar.gz files into:", RAW_DIR)
        return

    # idempotent extraction loop
    for arc in archives:
        flag = RAW_DIR / f".done_{arc.name}"
        if flag.exists():
            print(f"Already extracted: {arc.name}")
            continue
        if not arc.exists():
            print(f"Missing archive: {arc}")
            continue
        safe_extract(arc, RAW_DIR)
        flag.touch()
        open(flag, "w").close()

if __name__ == "__main__":
    main()