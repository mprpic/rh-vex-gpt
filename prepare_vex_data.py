#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
# ]
# ///

import sys
import argparse
import glob
import json
import os
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
EXTRACTED_DIR = DATA_DIR / "extracted"
FORMATTED_DIR = DATA_DIR / "formatted"
Path(FORMATTED_DIR).mkdir(parents=True, exist_ok=True)

test_files = [
    "data/extracted/2019/cve-2019-12735.json",
    "data/extracted/2023/cve-2023-5344.json",
    "data/extracted/2021/cve-2021-34558.json",
    "data/extracted/2018/cve-2018-18495.json",
    "data/extracted/2016/cve-2016-0721.json",
]


class InvalidVEXException(ValueError):
    pass


class VEXParser:
    def __init__(self, file, json_data):
        self.file = file
        self.data = json.loads(json_data)
        self._validate()
        # We always have only one vulnerability object in the `vulnerabilities` array
        self.vulnerability = self.data["vulnerabilities"][0]
        self.product_tree = self.data["product_tree"]
        self.document_metadata = self.data["document"]

        # TODO: recognize unaffected CVE and "finish" early

        self.cve = self.extract_cve_id()
        self.summary = self.extract_summary()
        self.description = self.extract_description()
        self.statement = self.extract_statement()
        self.cwe = self.extract_cwe_id()
        self.acknowledgments = self.extract_acknowlegments()

        # TODO:
        # CWE ID
        # Exploit
        # Products and components
        # acks
        # References
        # CVSS
        # impact
        # mitigation  - get from remediations
        # public
        # discovered
        #

    def _validate(self) -> None:
        """Check that this is a valid VEX file with all the data we expect it to have"""
        if "vulnerabilities" not in self.data or len(self.data["vulnerabilities"]) == 0:
            raise InvalidVEXException(f"No vulnerabilities object found in {self.file}")

        if "product_tree" not in self.data or "branches" not in self.data["product_tree"]:
            raise InvalidVEXException(f"No product tree or product branches found in  {self.file}")

        if "document" not in self.data:
            raise InvalidVEXException(f"No document object found in {self.file}")

    def extract_summary(self) -> None | str:
        if "notes" not in self.vulnerability:
            return None
        return next(
            (
                note.get("text")
                for note in self.vulnerability["notes"]
                if note.get("category") == "summary"
            ),
            None,
        )

    def extract_description(self) -> None | str:
        if "notes" not in self.vulnerability:
            return None
        return next(
            (
                note.get("text")
                for note in self.vulnerability["notes"]
                if note.get("category") == "description"
            ),
            None,
        )

    def extract_statement(self) -> None | str:
        if "notes" not in self.vulnerability:
            return None
        return next(
            (
                note.get("text")
                for note in self.vulnerability["notes"]
                if note.get("category") == "other" and note.get("title") == "Statement"
            ),
            None,
        )

    def extract_acknowlegments(self):
        if "acknowledgments" not in self.vulnerability:
            return
        acknowledgments = []
        for ack in self.vulnerability["acknowledgments"]:
            acknowledgments.append(ack.get("text"))

    def extract_cve_id(self) -> None | str:
        return self.vulnerability.get("cve")

    def extract_cwe_id(self) -> None | str:
        cwe = self.vulnerability.get("cwe")
        if cwe:
            return f"{cwe['id']} ({cwe['name']})"
        return cwe

    def print(self) -> None:
        print("BEGIN_VULNERABILITY")
        print("CVE ID:", self.cve)
        print("Summary:", self.summary)
        print("Description:", self.description)
        print("CWE:", self.cwe)
        print("END_VULNERABILITY")

    def save_to_files(self):
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Process VEX files to extract vulnerability information"
    )
    parser.add_argument(
        "--path",
        "-p",
        help="Path to a specific VEX file or directory containing VEX files",
    )
    parser.add_argument("--print", "-o", action="store_true", help="Print generated content")

    args = parser.parse_args()

    if args.path:
        input_path = args.path
        if os.path.isfile(input_path):
            files = [input_path]
        elif os.path.isdir(input_path):
            files = glob.glob(os.path.join(input_path, "**/*.json"), recursive=True)
        else:
            files = []
        if not files:
            print(f"No file(s) found at: {input_path}")
            sys.exit(1)
    else:
        files = glob.glob(str(EXTRACTED_DIR / "**/*.json"), recursive=True)
        if not files:
            print("No VEX files found in the input directory.")
            sys.exit(1)

    files = test_files
    print(f"Preparing data from {len(files)} files")

    for file in files:
        with open(file) as f:
            json_data = f.read()
        vex = VEXParser(file, json_data)

        if args.print:
            vex.print()
        else:
            vex.save_to_files()

    print("Done!")


if __name__ == "__main__":
    main()
