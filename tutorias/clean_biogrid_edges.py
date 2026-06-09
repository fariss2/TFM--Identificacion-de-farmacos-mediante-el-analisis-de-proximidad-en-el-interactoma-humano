import argparse
import csv
import re
from pathlib import Path


UNIPROT_RE = re.compile(
    r"^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})$"
)


def split_identifier(value):
    value = value.strip()
    if value in {"", "-"}:
        return []
    return [part.strip() for part in value.split("|") if part.strip() and part.strip() != "-"]


def valid_uniprot(value):
    return bool(UNIPROT_RE.match(value))


def clean_edges(input_path, output_path, report_path):
    counters = {
        "input_rows": 0,
        "expanded_rows": 0,
        "removed_missing_id": 0,
        "removed_invalid_uniprot": 0,
        "removed_self_loop": 0,
        "removed_duplicate_or_inverse": 0,
        "kept_edges": 0,
    }

    seen_edges = set()

    with input_path.open(newline="", encoding="utf-8") as in_fh, output_path.open(
        "w", newline="", encoding="utf-8"
    ) as out_fh:
        reader = csv.DictReader(in_fh)
        if "source" not in reader.fieldnames or "target" not in reader.fieldnames:
            raise ValueError("El CSV de entrada debe contener columnas 'source' y 'target'.")

        writer = csv.DictWriter(out_fh, fieldnames=["source", "target"])
        writer.writeheader()

        for row in reader:
            counters["input_rows"] += 1
            sources = split_identifier(row["source"])
            targets = split_identifier(row["target"])

            if not sources or not targets:
                counters["removed_missing_id"] += 1
                continue

            for source in sources:
                for target in targets:
                    counters["expanded_rows"] += 1

                    if not valid_uniprot(source) or not valid_uniprot(target):
                        counters["removed_invalid_uniprot"] += 1
                        continue

                    if source == target:
                        counters["removed_self_loop"] += 1
                        continue

                    edge_key = tuple(sorted((source, target)))
                    if edge_key in seen_edges:
                        counters["removed_duplicate_or_inverse"] += 1
                        continue

                    seen_edges.add(edge_key)
                    writer.writerow({"source": source, "target": target})
                    counters["kept_edges"] += 1

    with report_path.open("w", encoding="utf-8") as report_fh:
        report_fh.write("Filtro de limpieza de BioGRID\n")
        report_fh.write(f"Entrada: {input_path}\n")
        report_fh.write(f"Salida: {output_path}\n\n")
        for key, value in counters.items():
            report_fh.write(f"{key}: {value}\n")


def parse_args():
    repo_root = Path(__file__).resolve().parent
    default_input = repo_root / "data" / "biogrid_edges.csv"
    default_output = repo_root / "data" / "biogrid_edges_clean.csv"
    default_report = repo_root / "data" / "biogrid_edges_clean_report.txt"

    parser = argparse.ArgumentParser(
        description="Limpia aristas BioGRID: elimina '-', expande IDs con '|', valida UniProt y quita duplicados inversos."
    )
    parser.add_argument("--input", type=Path, default=default_input, help="CSV de entrada.")
    parser.add_argument("--output", type=Path, default=default_output, help="CSV limpio de salida.")
    parser.add_argument("--report", type=Path, default=default_report, help="Informe de filtros aplicados.")
    return parser.parse_args()


def main():
    args = parse_args()
    clean_edges(args.input, args.output, args.report)


if __name__ == "__main__":
    main()
