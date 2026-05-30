#!/usr/bin/env python
"""
Check that all files from the reference/ folder are present in results/ folder
after running the msdft.py script. For some files the contents are also compared.
"""
import pathlib
import pandas as pd

def compare_file_content(path_ref, path):
    suffix = path_ref.suffix
    if path_ref.is_dir():
        return
    print(f"   * comparing contents of {path} with {path_ref} ... ", end="")
    match suffix:
        case ".csv":
            df_ref = pd.read_csv(path_ref)
            df = pd.read_csv(path)
            pd.testing.assert_frame_equal(df, df_ref, atol=1.0e-5, rtol=1.0e-3)
        case _:
            # ignore all other file types
            print("skipped ", end="")
    print("✓")


class MissingOutputException(Exception):
    pass

for filename_ref in pathlib.Path("reference").rglob("*"):
    path_ref = pathlib.Path(filename_ref)
    path = pathlib.Path("results") / path_ref.relative_to("reference")
    if not path.exists():
        raise MissingOutputException(
            f"  Expected output path '{path}' is missing, the reference path is '{path_ref}'"
        )
    compare_file_content(path_ref, path)
print("passed")
