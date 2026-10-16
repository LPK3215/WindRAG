"""支持 python -m pure_rag <子命令>"""

import sys

from pure_rag.cli import main

if __name__ == "__main__":
    sys.exit(main())
