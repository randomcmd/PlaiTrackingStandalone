# https://stackoverflow.com/questions/75048986/way-to-temporarily-change-the-directory-in-python-to-execute-code-without-affect
import contextlib
import os
import sys


@contextlib.contextmanager
def model_context(working_directory):
    old_working_directory = os.getcwd()

    # This could raise an exception, but it's probably
    # best to let it propagate and let the caller
    # deal with it, since they requested x
    os.chdir(working_directory)

    # To avoid having to fork the dependencies or removing the print statements I redirect stdout to stderr for this
    stdout = sys.stdout
    sys.stdout = sys.stderr

    try:
        yield

    finally:
        # This could also raise an exception, but you *really*
        # aren't equipped to figure out what went wrong if the
        # old working directory can't be restored.
        os.chdir(old_working_directory)
        sys.stdout = stdout