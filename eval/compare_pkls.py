import argparse
import json
from pathlib import Path

# Fix python path
import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.g1_metrics import load_g1_motion, evaluate_g1_fk_metrics

def main(ref_pkl, exec_pkl):
    ref_motion = load_g1_motion(ref_pkl)
    exec_motion = load_g1_motion(exec_pkl)
    
    # We can inject audio_path manually if not present, but let's assume it was saved.
    # If not, the BAP/BAS metrics will just be None, but we mainly care about FK metrics (Sliding, etc)
    model_path = "third_party/unitree_g1_description/g1_29dof_rev_1_0.xml"
    
    print("Evaluating Reference Motion (Diffusion)...")
    ref_metrics = evaluate_g1_fk_metrics(ref_motion, model_path, root_quat_order='xyzw')
    
    print("Evaluating Executed Motion (SONIC)...")
    # For SONIC, the motion is actual executed, so its FPS might have minor jitter, but we set it to 30.
    exec_metrics = evaluate_g1_fk_metrics(exec_motion, model_path, root_quat_order='xyzw')
    
    print("\n" + "="*50)
    print(f"{'Metric':<25} | {'Diffusion':<10} | {'SONIC Exec':<10}")
    print("-" * 50)
    
    keys_to_compare = [
        "G1FKBAS", "G1FKRoboPerformBAS", "G1BeatF1", 
        "G1FootSliding", "G1GroundPenetration", "G1FootClearanceMean"
    ]
    
    for k in keys_to_compare:
        ref_val = ref_metrics.get(k, 0.0) if ref_metrics else 0.0
        exec_val = exec_metrics.get(k, 0.0) if exec_metrics else 0.0
        print(f"{k:<25} | {ref_val:<10.4f} | {exec_val:<10.4f}")
    
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True)
    parser.add_argument("--exe", required=True)
    args = parser.parse_args()
    main(args.ref, args.exe)
