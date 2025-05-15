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
    "data/extracted/2022/cve-2022-3723.json",
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
        self.cve = self.extract_cve_id()
        self.summary = self.extract_summary()
        self.description = self.extract_description()
        self.statement = self.extract_statement()
        self.cwe = self.extract_cwe_id()
        self.acknowledgments = self.extract_acknowledgments()
        self.exploit_exists = self.extract_exploit()
        self.mitigation = self.extract_mitigation()
        self.references = self.extract_references()
        self.discovered_dt = self.extract_discovered_date()
        self.public_date = self.extract_public_date()

        self.products, self.components = self.extract_products_and_components()
        self.relationships = self.extract_relationship_ids()
        self.rel_to_impact = self.extract_impact()
        self.rel_to_cvss = self.extract_cvss()
        # Products and components
        # CVSS - per product
        # impact  - get form threats
        # mitigation  - get from remediations
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

    def extract_acknowledgments(self) -> None | str:
        if "acknowledgments" not in self.vulnerability:
            return None
        acks = []
        for ack in self.vulnerability["acknowledgments"]:
            ack_lines = []

            if "names" in ack and ack["names"]:
                ack_lines.append(f"  Contributor: {'; '.join(ack['names'])}")
            if "organization" in ack and ack["organization"]:
                ack_lines.append(f"  Organization: {ack['organization']}")
            if "summary" in ack and ack["summary"]:
                ack_lines.append(f"  Details: {ack['summary']}")

            acks.append("- " + ack_lines[0][2:])
            acks.extend(ack_lines[1:])

        return "\n".join(acks) if acks else None

    def extract_cve_id(self) -> None | str:
        return self.vulnerability.get("cve")

    def extract_cwe_id(self) -> None | str:
        cwe = self.vulnerability.get("cwe")
        if cwe:
            return f"{cwe['id']} ({cwe['name']})"
        return cwe

    def extract_exploit(self) -> bool:
        return any(
            threat["category"] == "exploit_status"
            for threat in self.vulnerability.get("threats", [])
        )

    def extract_discovered_date(self) -> None | str:
        return self.vulnerability.get("discovery_date")

    def extract_public_date(self) -> None | str:
        return self.vulnerability.get("release_date")

    def extract_references(self) -> None | str:
        if "references" not in self.vulnerability:
            return None
        refs = []
        for ref in self.vulnerability["references"]:
            if ref["summary"] == ref["url"]:
                refs.append(f"- External: {ref['url']}")
            else:
                refs.append(f"- {ref['summary']}: {ref['url']}")

        return "\n".join(refs)

    def extract_mitigation(self) -> None | str:
        for remediation in self.vulnerability.get("remediations", []):
            # Only one mitigation can exist and it applies to all products.
            if remediation["category"] == "workaround":
                return remediation["details"]
        return None

    def extract_products_and_components(self) -> (dict, dict):
        product_id_to_cpe = {}
        component_id_to_cpe = {}

        def process_branch(branch):
            if "product" in branch:
                product = branch["product"]
                product_id = product.get("product_id")

                if product_id and "product_identification_helper" in product:
                    helper = product["product_identification_helper"]
                    if "cpe" in helper:
                        product_id_to_cpe[product_id] = {
                            "cpe": helper["cpe"],
                            "name": product["name"],
                        }
                    if "purl" in helper:
                        component_id_to_cpe[product_id] = helper["purl"]

            # Process any sub-branches recursively
            if "branches" in branch:
                for sub_branch in branch["branches"]:
                    process_branch(sub_branch)

        # Process all top-level branches
        for branch in self.product_tree.get("branches", []):
            process_branch(branch)

        return product_id_to_cpe, component_id_to_cpe

    def extract_relationship_ids(self) -> list:
        relationship_ids = []
        for rel in self.product_tree.get("relationships"):
            relationship_ids.append(rel["full_product_name"]["product_id"])
        return relationship_ids

    def extract_impact(self) -> dict:
        rel_to_impact = {}
        for threat in self.vulnerability.get("threats", []):
            if threat["category"] == "impact":
                for product_id in threat["product_ids"]:
                    rel_to_impact[product_id] = threat["details"]
        return rel_to_impact

    def extract_cvss(self) -> dict:
        rel_to_cvss = {}
        for score in self.vulnerability.get("scores", []):
            if "cvss_v2" in score:
                cvss = {
                    "score": score["cvss_v2"]["baseScore"],
                    "vector": score["cvss_v2"]["vectorString"],
                }
            elif "cvss_v3" in score:
                cvss = {
                    "score": score["cvss_v3"]["baseScore"],
                    "vector": score["cvss_v3"]["vectorString"].removeprefix("CVSS:3.1/"),
                }
            else:
                raise ValueError(f"Unrecognized score type in {self.file}: {score}")
            for rel_id in score["products"]:
                rel_to_cvss[rel_id] = cvss
        return rel_to_cvss

    def print_text(self) -> None:
        print("BEGIN_VULNERABILITY")
        print("CVE ID:", self.cve)
        if self.discovered_dt:
            print("Discovered date:", self.discovered_dt)
        print("Public date:", self.public_date)
        if self.summary:
            print("Summary:", self.summary)
        if self.description:
            print("Description:", self.description)
        if self.mitigation:
            print("Mitigation:", self.mitigation)
        if self.statement:
            print("Statement:", self.statement)
        if self.references:
            print("References:\n" + self.references)
        if self.cwe:
            print("CWE:", self.cwe)
        if self.acknowledgments:
            print("Acknowledgments:\n" + self.acknowledgments)
        print(f"Exploit exists: {self.exploit_exists}")
        from pprint import pprint

        # pprint(self.products)
        # pprint(self.components)
        # pprint(self.relationships)
        # pprint(self.rel_to_impact)
        # pprint(self.rel_to_cvss)
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

        if '"cpe:/a:redhat"' in json_data:
            # Skip VEX files for CVEs that do not affect Red Hat products.
            continue

        vex = VEXParser(file, json_data)
        if args.print:
            vex.print_text()
        else:
            vex.save_to_files()

    print("Done!")


if __name__ == "__main__":
    main()
