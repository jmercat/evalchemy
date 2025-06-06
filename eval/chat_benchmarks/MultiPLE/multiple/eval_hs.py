from pathlib import Path
from .safe_subprocess import run


def eval_script(path: Path):
    r = run(["runghc", str(path)])
    if r.timeout:
        status = "Timeout"
    elif r.exit_code == 0:
        status = "OK"
    elif "Syntax error":
        status = "SyntaxError"
    else:
        status = "Exception"
    return {
        "status": status,
        "exit_code": r.exit_code,
        "stdout": r.stdout,
        "stderr": r.stderr,
    }
