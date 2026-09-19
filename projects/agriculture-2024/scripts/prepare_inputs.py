"""Verify official local attachments and rebuild the curated basis without copying raw files."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from q1_data import read_crops, read_parameters, read_plantings_2023, read_plots, sha256
from portable_data import basis_payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attachment-1", type=Path, default=ROOT / ".local-inputs" / "附件1.xlsx")
    parser.add_argument("--attachment-2", type=Path, default=ROOT / ".local-inputs" / "附件2.xlsx")
    args = parser.parse_args()
    expected = json.loads((ROOT / "data" / "input_basis.json").read_text(encoding="utf-8"))
    hashes = {"attachment_1": sha256(args.attachment_1), "attachment_2": sha256(args.attachment_2)}
    if hashes != expected["source_hashes"]:
        raise SystemExit("Input SHA-256 differs from the frozen study. Check the source; no basis was replaced.")
    plots = read_plots(args.attachment_1)
    crops = read_crops(args.attachment_1)
    parameters = read_parameters(args.attachment_2, crops)
    plantings = read_plantings_2023(args.attachment_2, plots, crops)
    sales = defaultdict(float)
    for row in plantings:
        parameter = parameters[(row.crop_id, plots[row.plot].land_type, row.season)]
        sales[(row.crop_id, row.season)] += row.area_mu * parameter.yield_jin_per_mu
    order = {"单季": 0, "第一季": 1, "第二季": 2}
    sales = dict(sorted(sales.items(), key=lambda item: (item[0][0], order[item[0][1]])))
    rebuilt = basis_payload(plots, crops, parameters, plantings, sales, hashes)
    if rebuilt != expected:
        raise SystemExit("Reconstructed fields differ from the frozen derived basis; review required.")
    output = ROOT / ".reproduction" / "input_basis_rebuilt.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(rebuilt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "fields_equal": True, "plots": len(plots), "crops": len(crops), "area_mu": sum(p.area_mu for p in plots.values())}))


if __name__ == "__main__":
    main()
