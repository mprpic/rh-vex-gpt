#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#     "tqdm>=4.61.0"
# ]
# ///
import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

DATA_DIR = Path(__file__).parent / "data"
EXTRACTED_DIR = DATA_DIR / "extracted"
FORMATTED_DIR = DATA_DIR / "formatted"
AGGREGATED_FILE = FORMATTED_DIR / "vex_data.jsonl"
Path(FORMATTED_DIR).mkdir(parents=True, exist_ok=True)

test_files = [
    "data/extracted/2019/cve-2019-12735.json",
    "data/extracted/2023/cve-2023-5344.json",
    "data/extracted/2021/cve-2021-34558.json",
    "data/extracted/2018/cve-2018-18495.json",
    "data/extracted/2016/cve-2016-0721.json",
    "data/extracted/2022/cve-2022-3723.json",
    "data/extracted/2024/cve-2024-53907.json",
    "data/extracted/2025/cve-2025-46420.json",
    "data/extracted/2025/cve-2025-46727.json",
    "data/extracted/2025/cve-2025-37867.json",
    "data/extracted/2025/cve-2025-3522.json",
    "data/extracted/2025/cve-2025-2912.json",
]

PRODUCT_STATUS_MAP = {
    "known_not_affected": "Not Affected",
    "known_affected": "Affected",
    "under_investigation": "Under Investigation",
    "fixed": "Fixed",
}

RESOLUTION_MAP = {"Affected": "Fix not yet available"}


class InvalidVEXException(ValueError):
    pass


class VEXParser:
    def __init__(self, file, json_data):
        self.file = file
        self.data = json_data
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
        self.rel_to_affectedness = self.extract_product_status()
        self.rel_to_remediation = self.extract_remediations()

        self.product_map = self.generate_product_map()

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

    def extract_products_and_components(self) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
        product_data: dict[str, dict[str, str]] = {}
        component_data: dict[str, str] = {}

        def normalize_rhel_product_name(cpe: str, product_id: str, product_name: str) -> str:
            """Arbitrary rules to convert the myriad of ways RHEL is named into something more
            normalized and readable."""
            # Ignore known bad CPE for RHEL SAM
            if "rhel_sam" in cpe:
                return product_name

            # Check if we already have a good enough product name and version
            if not re.match(r"^Red Hat Enterprise Linux \d{1,2}(\.\d{1,2})?$", product_name):
                # If not, find the version from the product ID.
                if version := re.search(r"(\d{1,2}\.\d{1,2})", product_id):
                    return f"Red Hat Enterprise Linux {version.group(1)}"

            return product_name

        def process_branch(branch: dict[str, Any]) -> None:
            if "product" in branch:
                product = branch["product"]
                product_id = product.get("product_id")

                if "product_identification_helper" in product:
                    helper = product["product_identification_helper"]
                    if "cpe" in helper:
                        cpe = helper["cpe"]
                        product_name = product["name"]
                        if (":rhel_" in cpe) or (":enterprise_linux:" in cpe):
                            product_name = normalize_rhel_product_name(
                                cpe, product_id, product_name
                            )

                        product_data[product_id] = {
                            "cpe": cpe,
                            "name": product_name,
                        }
                    if "purl" in helper:
                        component_data[product_id] = helper["purl"]

                if (
                    branch["category"] == "product_version"
                    and "product_identification_helper" not in product
                ):
                    # E.g. 2021/cve-2021-34558.json, 2025/cve-2025-25208.json
                    component_data[product_id] = "N/A"

            # Process any sub-branches recursively
            if "branches" in branch:
                for sub_branch in branch["branches"]:
                    process_branch(sub_branch)

        # Process all top-level branches
        for b in self.product_tree.get("branches", []):
            process_branch(b)

        return product_data, component_data

    def extract_relationship_ids(self) -> list:
        relationship_ids = []
        for rel in self.product_tree.get("relationships", []):
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
            for version_obj in ("cvss_v2", "cvss_v3"):
                cvss_obj = score.get(version_obj)
                if not cvss_obj:
                    continue
                cvss = (
                    f"CVSSv{cvss_obj['version']}: {cvss_obj['baseScore']} / "
                    f"{cvss_obj['vectorString'].removeprefix('CVSS:3.1/').removeprefix('CVSS:3.0/')}"
                )
                break
            else:
                raise ValueError(f"Unrecognized score type in {self.file}: {score}")

            for rel_id in score["products"]:
                rel_to_cvss[rel_id] = cvss
        return rel_to_cvss

    def extract_remediations(self) -> dict:
        rel_to_remediations = {}
        for remediation in self.vulnerability.get("remediations", []):
            # Skip workarounds (mitigations); extracted in extract_mitigation(). Skip vendor
            # fixes too because we have that same information from product_status.
            if remediation["category"] == "workaround" or remediation["category"] == "vendor_fix":
                continue
            for product_id in remediation["product_ids"]:
                rel_to_remediations[product_id] = RESOLUTION_MAP.get(
                    remediation["details"], remediation["details"]
                )
        return rel_to_remediations

    def extract_product_status(self) -> dict:
        rel_to_affectedness = {}
        for status, product_ids in self.vulnerability.get("product_status", {}).items():
            for product_id in product_ids:
                rel_to_affectedness[product_id] = PRODUCT_STATUS_MAP[status]
        return rel_to_affectedness

    def generate_product_map(self) -> dict:
        product_map = {}

        def add_to_product_map(rel_id, product_name, product_cpe, component_purl):
            cvss = self.rel_to_cvss.get(rel_id) or "N/A"
            impact = self.rel_to_impact.get(rel_id, "None")
            status = self.rel_to_affectedness[rel_id]
            if rel_id in self.rel_to_remediation:
                status += f" / {self.rel_to_remediation[rel_id]}"

            if product_name not in product_map:
                product_map[product_name] = {
                    "cpes": {product_cpe},
                    "components": {
                        component_purl: {"status": status, "cvss": cvss, "impact": impact}
                    },
                }

            else:
                product_map[product_name]["cpes"].add(self.products[product_id]["cpe"])
                product_map[product_name]["components"][component_purl] = {
                    "status": status,
                    "cvss": cvss,
                    "impact": impact,
                }

        if not self.relationships:
            for product_id, product_data in self.products.items():
                product_name = product_data["name"]
                product_cpe = product_data["cpe"]
                add_to_product_map(product_id, product_name, product_cpe, "N/A")
            return product_map

        for rel_id in self.relationships:
            product_id, component_id = rel_id.split(":", maxsplit=1)

            product_data = self.products[product_id]
            product_name = product_data["name"]
            product_cpe = product_data["cpe"]

            if component_id not in self.components:
                # Let's try splitting again in case this ID consists of an RPM module and RPM
                parts = component_id.split(":")
                rpm_mod_id, comp_id = ":".join(parts[0:4]), ":".join(parts[4:])

                if rpm_mod_id not in self.components:
                    print(f"ERROR: Missing RPM module component {rpm_mod_id} in {self.file}")
                    continue
                if comp_id not in self.components:
                    print(f"WARNING: Missing RPM component {comp_id} in {self.file}")
                    continue

                rpm_mod_purl = self.components[rpm_mod_id]
                add_to_product_map(rel_id, product_name, product_cpe, rpm_mod_purl)

                comp_purl = self.components[comp_id]
                add_to_product_map(rel_id, product_name, product_cpe, comp_purl)
            else:
                # Some components that may be missing a product helper identifier are not
                # included in our components map so should be skipped here as well.
                if component_id not in self.components:
                    continue
                comp_purl = self.components[component_id]
                add_to_product_map(rel_id, product_name, product_cpe, comp_purl)

        return product_map

    def create_documents(self) -> dict[str, str]:
        documents = {}
        for product, data in self.product_map.items():
            output = ["BEGIN_VULNERABILITY", f"CVE ID: {self.cve}"]
            if self.discovered_dt:
                output.append(f"Discovered date: {self.discovered_dt}")
            output.append(f"Public date: {self.public_date}")
            if self.summary:
                output.append(f"Summary: {self.summary}")
            if self.description:
                output.append(f"Description: {self.description}")
            if self.mitigation:
                output.append(f"Mitigation: {self.mitigation}")
            if self.statement:
                output.append(f"Statement: {self.statement}")
            if self.references:
                output.append(f"References:\n{self.references}")
            if self.cwe:
                output.append(f"CWE: {self.cwe}")
            if self.acknowledgments:
                output.append(f"Acknowledgments:\n{self.acknowledgments}")
            output.append(f"Exploit exists: {self.exploit_exists}")
            output.append(f"Product: {product} ({', '.join(data['cpes'])})")
            for component, comp_data in data["components"].items():
                output.append(
                    f"- {comp_data['status']}: {component} "
                    f"(Impact: {comp_data['impact']}, {comp_data['cvss']})"
                )
            output.append("END_VULNERABILITY")

            # Create a file name from the CVE ID and product name
            product_part = product.lower().replace(" ", "_").replace("/", "_")
            file_name = f"{self.cve}:{product_part}"
            documents[file_name] = "\n".join(output)

        return documents

    def save_to_files(self):
        documents = self.create_documents()
        for file_name, content in documents.items():
            # Create the output file with .txt extension
            output_path = FORMATTED_DIR / f"{file_name}.txt"
            with open(output_path, "w") as f:
                f.write(content)


def aggregate_data() -> None:
    formatted_files = list(FORMATTED_DIR.glob("*.txt"))
    if not formatted_files:
        print(f"No .txt files found in {FORMATTED_DIR}")
        return

    print(f"Found {len(formatted_files)} files to aggregate into a JSONL file")
    with open(AGGREGATED_FILE, "w+") as outfile:
        for file_path in tqdm(formatted_files, "Aggregating files"):
            with open(file_path) as infile:
                content = infile.read().strip()
                record = {"text": content}
                json.dump(record, outfile, ensure_ascii=False)
                outfile.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Process VEX files to extract vulnerability information"
    )
    parser.add_argument(
        "--path",
        "-p",
        help=(
            f"Path to a specific file or directory "
            f"(defaults to {EXTRACTED_DIR} if option not used)"
        ),
    )
    parser.add_argument("--print", "-o", action="store_true", help="Print generated content")
    parser.add_argument(
        "--force-refresh",
        "-f",
        action="store_true",
        help="Process all files regardless of timestamps",
    )

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
        print(f"Searching for relevant files in {EXTRACTED_DIR}")
        all_files = glob.glob(str(EXTRACTED_DIR / "**/*.json"), recursive=True)
        if not all_files:
            print("No VEX files found in the input directory")
            sys.exit(1)
        print(f"Filtering {len(all_files)} files based on modified time")

        # Filter files based on timestamp unless force-refresh is specified
        if args.force_refresh:
            files = all_files
        else:
            files = []
            all_formatted_files = glob.glob(str(FORMATTED_DIR / "*.txt"))
            formatted_files_by_cve = defaultdict(list)
            for ff in all_formatted_files:
                # Get CVE ID from filename: CVE-2025-0514:red_hat_enterprise_linux_9.txt
                cve_id, _, _ = Path(ff).stem.partition(":")
                formatted_files_by_cve[cve_id].append(ff)

            for extracted_file in all_files:
                # Get CVE ID from filename: cve-2025-0001.json
                cve_id = Path(extracted_file).stem.upper()
                related_formatted_files = formatted_files_by_cve.get(cve_id, [])

                # If no formatted files exist, include this file for processing
                if not related_formatted_files:
                    files.append(extracted_file)
                    continue

                # Get the most recent formatted file timestamp
                extracted_mtime = os.path.getmtime(extracted_file)
                newest_formatted_mtime = max(os.path.getmtime(f) for f in related_formatted_files)

                # Include file if extracted version is newer than formatted version
                if extracted_mtime > newest_formatted_mtime:
                    files.append(extracted_file)

    if not files:
        print("No files to process")
    else:
        print(f"Preparing data from {len(files)} files")

        iterator = files if args.print else tqdm(files, "Processing VEX files")
        for file in iterator:
            with open(file) as f:
                content = f.read()

            json_data = json.loads(content)
            vex = VEXParser(file, json_data)
            if args.print:
                for doc in vex.create_documents().values():
                    print(doc)
            else:
                vex.save_to_files()

    if not args.print:
        print("Aggregating data into a JSONL file")
        aggregate_data()

    print("Done!")


if __name__ == "__main__":
    main()
