import os
import shutil
import subprocess


def get_git_tracked_files(src_dir):
    """
    Returns a list of all files tracked by Git in the given directory.
    Returns None if git command fails.
    """
    try:
        # The command to list all tracked files, including submodules
        command = ["git", "ls-files", "--recurse-submodules"]
        # Execute the command in the source directory
        result = subprocess.run(
            command,
            cwd=src_dir,
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8",  # Explicitly set encoding
        )
        # Split the output into a list of files
        files = result.stdout.strip().split("\n")
        return [f for f in files if f]  # Filter out empty strings
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Could not list git tracked files: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while listing git files: {e}")
        return None


def copy_project(src, dst):
    """
    Copies the project from src to dst.
    It first tries to copy only git-tracked files.
    If that fails, it falls back to copying all files and ignoring a predefined set of patterns.
    """
    if os.path.exists(dst):
        print(f"Destination directory {dst} already exists. Removing it.")
        shutil.rmtree(dst)

    os.makedirs(dst, exist_ok=True)
    print(f"Copying project from {src} to {dst}...")

    git_files = get_git_tracked_files(src)

    if git_files:
        print("Found git tracked files. Copying them...")
        for file_path in git_files:
            source_path = os.path.join(src, file_path)
            destination_path = os.path.join(dst, file_path)

            if os.path.exists(source_path):
                # Create the destination directory if it doesn't exist
                os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                # Copy the file
                shutil.copy2(source_path, destination_path)
        print("Project copied successfully using git tracked files.")
    else:
        print(
            "Fallback: Could not get git tracked files. Copying using ignore patterns."
        )
        # Fallback to the old method
        shutil.rmtree(dst)  # Clean up the empty dir created before
        ignore = shutil.ignore_patterns(
            ".git",
            ".venv",
            ".*",
            "*.pyc",
            "__pycache__",
            "test_logs",
            "memory_system/data",
        )
        shutil.copytree(src, dst, ignore=ignore)
        print("Project copied successfully using ignore patterns.")


if __name__ == "__main__":
    # Define source and destination directories
    source_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    destination_dir = "/tmp/gui-agent-memory-clone"

    copy_project(source_dir, destination_dir)

    # Create necessary empty directories in the clone that are in .gitignore
    print("Creating necessary empty directories in the destination...")
    os.makedirs(os.path.join(destination_dir, "test_logs"), exist_ok=True)
    os.makedirs(
        os.path.join(destination_dir, "memory_system/data/chroma"), exist_ok=True
    )
    # Create an empty .env file from .env.example
    env_example_path = os.path.join(destination_dir, ".env.example")
    env_path = os.path.join(destination_dir, ".env")
    if os.path.exists(env_example_path):
        shutil.copy2(env_example_path, env_path)
        print("Created .env file from .env.example.")

    print("Setup of the cloned project is complete.")
