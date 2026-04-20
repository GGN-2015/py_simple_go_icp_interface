import subprocess
import sys
from typing import List, Optional

def run_executable(
    exe_path: str,
    args: Optional[List[str]] = None,
    print_output: bool = True,
    check: bool = False
) -> int:
    """
    Cross-platform execution of external executable files,
    real-time output of stdout/stderr to the console
    
    Args:
        exe_path: Path to the executable (Windows: xxx.exe, Linux: ./xxx)
        args: List of command-line arguments, e.g., ["-i", "input.txt", "-o", "output.txt"]
        print_output: Whether to print program output to the console in real-time (default: True)
        check: Whether to raise an exception if the program returns a non-zero exit code (default: False)
    
    Returns:
        Return code of the executed command
    """
    # Build the full command: [executable path, arg1, arg2, ...]
    command = [exe_path]
    if args is not None:
        command.extend(args)
    
    try:
        # Core: Cross-platform compatibility + real-time output stream
        with subprocess.Popen(
            command,
            stdout=sys.stdout if print_output else subprocess.PIPE,
            stderr=sys.stderr if print_output else subprocess.PIPE,
            text=True,       # Text mode (automatically handles line breaks, no garbled characters)
            bufsize=1,       # Line buffering to ensure real-time output
            universal_newlines=True
        ) as proc:
            # Wait for the process to finish
            returncode = proc.wait()

        # Raise an exception for non-zero exit code if check is enabled
        if check and returncode != 0:
            raise subprocess.CalledProcessError(returncode, command)
        
        return returncode

    except FileNotFoundError:
        raise FileNotFoundError(f"Executable file not found: {exe_path}, please check if the path is correct")
    except Exception as e:
        raise RuntimeError(f"Failed to execute command: {str(e)}")


if __name__ == "__main__":
    from utils import GO_ICP_EXE
    run_executable(
        exe_path=GO_ICP_EXE,
        args=[]
    )
