import subprocess


def get_git_info():
    """Function retrieves current git info, i.e. SHA-1 hash and branch."""
    try:
        commit_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
        branch_name = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD']).strip().decode(
            'utf-8')
        return commit_hash, branch_name
    except Exception as e:
        print(f"Error fetching Git info: {e}")
        return None, None
