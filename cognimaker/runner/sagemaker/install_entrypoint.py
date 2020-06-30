import shutil
import sys
from pathlib import Path
from typing import Union


def install(directory: Union[Path, str, None]):
    module_dir = Path(__file__).parent
    static_dir = module_dir / 'static_files'
    target_dir = Path(directory) if directory is not None else Path.cwd()

    try:
        for file in static_dir.iterdir():
            if file.name != '__pycache__':
                shutil.copy(file, target_dir)
    except shutil.Error as e:
        print("Error copying entrypoint files!", file=sys.stderr)
        print(e, file=sys.stderr)
        sys.exit(-1)


if __name__ == '__main__':
    target = sys.argv[1] if len(sys.argv) == 2 else None
    install(target)
