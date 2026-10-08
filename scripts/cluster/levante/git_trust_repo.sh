# Source with the repo path: `source git_trust_repo.sh "${REPO}"`.
# A worktree whose shared .git belongs to another user makes job-side git
# refuse the repo ("dubious ownership"), so run manifests recorded commit "".
# Trust exactly that path for this process via git's environment config;
# never edits any git config file.
_gtr_n=${GIT_CONFIG_COUNT:-0}
export "GIT_CONFIG_KEY_${_gtr_n}=safe.directory" "GIT_CONFIG_VALUE_${_gtr_n}=${1:?repo path}"
export GIT_CONFIG_COUNT=$((_gtr_n + 1))
unset _gtr_n
