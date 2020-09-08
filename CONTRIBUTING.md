# Contributing Guide

Thank you for your interest in contributing to PyTorch C++ Tutorials! Before you begin writing code, it is important that you share your intention to contribute with the team, based on the type of contribution:

1. You want to propose a new feature and implement it.
    Post about your intended feature in an issue, and we shall discuss the design and implementation. Once we agree that the plan looks good, go ahead and implement it.
2. You want to implement a feature or bug-fix for an outstanding issue.
    Search for your issue in the PyTorch issue list.
    Pick an issue and comment that you'd like to work on the feature or bug-fix.
    If you need more context on a particular issue, please ask and we shall provide.

Once you implement and test your feature or bug-fix, please submit a Pull Request to https://github.com/prabhuomkar/pytorch-cpp.

## Developing PyTorch C++ Tutorials

To develop PyTorch C++ Tutorials on your machine, here are some tips:
1. Install latest CMake from [here](https://cmake.org/)
2. Clone the repository
```
git clone https://github.com/prabhuomkar/pytorch-cpp
cd pytorch-cpp
```
3. Follow the instructions given [here](https://github.com/prabhuomkar/pytorch-cpp#generate-build-system).

## Codebase Structure

- [docker](docker/) - Docker entrypoint and other docker related scripts
- [extern](extern/) - External third-party C++ libraries
- [notebooks](notebooks/) - Interactive tutorials are inside this directory
- [tutorials](tutorials/) - All set of tutorials are here
- [utils](utils/) - Utility functions live here

## CI Failure Tips

Once you submit a PR or push a new commit to a branch that is in an active PR, Travis CI jobs will be run automatically. Some of these may fail and you will need to find out why, by looking at the logs.

Some failures might be related to specific hardware or environment configurations. In this case, if the job is run by TravisCI, make sure you check logs and see the line of failure/error to identify the cause.

## Discussion

If you want to discuss something more in detail, contact: 
- Omkar Prabhu - [prabhuomkar](https://github.com/prabhuomkar)
- Markus Fleischhacker - [mfl28](https://github.com/mfl28)