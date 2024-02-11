# Contribution Guidelines

We are very happy that you have decided to help us democratize meta-reinforcement learning 
research and experimentation. We highly welcome:

- Tests 🥺
- Bug reports and pull requests with fixes
- Performance and speed optimizations
- Documentation and comments
- Additional rules and goals, game mechanics (without significant overhead)

Do not be afraid to ask questions and share new ideas about possible directions for XLand-MiniGrid!

## Contributing to the codebase

Contributing code is done through standard github methods:

1. Fork this repo
2. Make a change and commit your code
3. Submit a pull request. It will be reviewed by maintainers, and they'll give feedback or make requests as applicable

```commandline
git clone git@github.com:corl-team/xland-minigrid.git
cd xland-minigrid
pip install -e ".[dev]"
```

## Code style

We use awesome [Ruff](https://docs.astral.sh/ruff/) linter and formatter and [Pyright](https://microsoft.github.io/pyright/#/) for type checking. 
The CI will run several checks on the new code pushed to the repository.
These checks can also be run locally without waiting for the CI by following the steps below: 

1. install [pre-commit](https://pre-commit.com/#install)
2. install the Git hooks by running `pre-commit install`

Once those two steps are done, the Git hooks will be run automatically at
every new commit. The Git hooks can also be run manually with 
`pre-commit run --all-files`, and if needed they can be 
skipped (not recommended) with `git commit --no-verify`.

Be sure to run and fix all issues from the `pre-commit run --all-files` before the push!
If you want to see possible problems before pre-commit, you can run `ruff check --diff .` 
and `ruff format --check` to see exact linter and formatter suggestions and possible fixes. 
Similarly, run `pyright src/xminigrid` to see possible problems with type hints.

# License

All contributions will fall under the project's original license.