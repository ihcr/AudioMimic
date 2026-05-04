import argparse
import re
from pathlib import Path
from urllib.request import urlopen


UPSTREAM_BASE = "https://raw.githubusercontent.com/unitreerobotics/unitree_ros/master"
G1_ROOT = "robots/g1_description"
MODEL_NAME = "g1_29dof_rev_1_0.xml"


def referenced_mesh_files(xml_text):
    return sorted(set(re.findall(r'<mesh[^>]+file="([^"]+)"', xml_text)))


def download_text(url):
    with urlopen(url, timeout=60) as response:
        return response.read().decode("utf-8")


def download_bytes(url):
    with urlopen(url, timeout=60) as response:
        return response.read()


def write_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def write_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def fetch_unitree_g1_description(output_root):
    output_root = Path(output_root)
    model_url = f"{UPSTREAM_BASE}/{G1_ROOT}/{MODEL_NAME}"
    readme_url = f"{UPSTREAM_BASE}/{G1_ROOT}/README.md"
    license_url = f"{UPSTREAM_BASE}/LICENSE"

    model_xml = download_text(model_url)
    write_text(output_root / MODEL_NAME, model_xml)
    write_text(output_root / "README.md", download_text(readme_url))
    write_text(output_root / "LICENSE", download_text(license_url))

    mesh_files = referenced_mesh_files(model_xml)
    for mesh_file in mesh_files:
        mesh_url = f"{UPSTREAM_BASE}/{G1_ROOT}/meshes/{mesh_file}"
        write_bytes(output_root / "meshes" / mesh_file, download_bytes(mesh_url))

    write_text(
        output_root / "SOURCE.txt",
        "\n".join(
            [
                "Repository: https://github.com/unitreerobotics/unitree_ros",
                f"Raw base: {UPSTREAM_BASE}",
                f"Model: {G1_ROOT}/{MODEL_NAME}",
                f"Meshes: {len(mesh_files)}",
                "",
            ]
        ),
    )
    return {
        "output_root": str(output_root),
        "model": str(output_root / MODEL_NAME),
        "mesh_count": len(mesh_files),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_root",
        default="third_party/unitree_g1_description",
        type=str,
    )
    return parser.parse_args()


def main():
    result = fetch_unitree_g1_description(parse_args().output_root)
    print(f"Saved Unitree G1 model to {result['model']}")
    print(f"Saved {result['mesh_count']} mesh files")


if __name__ == "__main__":
    main()
