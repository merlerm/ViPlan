# Coding Conventions

## Logging

We use the [mloggers](https://github.com/serhez/mloggers/) library in the project, developed by an Aalto colleague. Refer to the original repository for more information if needed. 
In the project, there is a wrapper for the logger under `code_helpers.py`, which can be imported using:

```from viplan.code_helpers import get_logger```.

The function takes a `log_level` string argument, which can be one of "debug", "info", "warning" or "error and represents the level of logging, and a `log_file` string argument, which is the path to the log file. If the `log_file` parameter is `None`, only console logging will be used.

As a general best practice, one should always create one instance of the logger at the beginning of each experiment/script, and then pass it as an argument to all functions that require logging. All models under `models/` can take the logger as input. Make sure to use the various logging levels appropriately (e.g.`logger.info` for generic information that should always be printed, but `logger.debug` for more verbose information that is not always required).