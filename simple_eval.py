import argparse
from trackeval.mot_utils import load_and_preprocess_data
from trackeval.metrics import HOTA, CLEAR, Identity, Count, VACE
import os
import numpy as np


def evaluate_mot(gt_path, tracker_path, metrics=['HOTA'], output_dir=None, create_plots=False, tracker_name=None):
    """Evaluates a tracker's output against ground truth (MOTChallenge format)."""

    if tracker_name is None:
        tracker_name = os.path.splitext(os.path.basename(tracker_path))[0]

    raw_data = load_and_preprocess_data(gt_path, tracker_path)

    metric_instances = []
    for metric_name in metrics:
        if metric_name.lower() == 'hota':
            metric_instances.append(HOTA())
        elif metric_name.lower() == 'clear':
            metric_instances.append(CLEAR())
        elif metric_name.lower() == 'identity':
            metric_instances.append(Identity())
        elif metric_name.lower() == 'count':
            metric_instances.append(Count())
        elif metric_name.lower() == 'vace':
            metric_instances.append(VACE())
        else:
            raise ValueError(f"Unsupported metric: {metric_name}")

    results = {}
    for metric in metric_instances:
        results[metric.get_name()] = metric.eval_sequence(raw_data)

    print("\n" + "="*60)
    print(" "*20 + "EVALUATION RESULTS" + " "*20)
    print("="*60)
    
    for metric_name, result in results.items():
        metric = next(m for m in metric_instances if m.get_name() == metric_name)
        
        summary = metric.summary_results({'COMBINED_SEQ': result})
        detailed = metric.detailed_results({'COMBINED_SEQ': result})
        
        print(f"\n{metric_name} Summary Results".center(60))
        print("="*60)
        for field, value in summary.items():
            if isinstance(value, (int, float)):
                print(f"{field:.<50}{value:>8.4f}")
            else:
                print(f"{field:.<50}{str(value):>8}")
            
        print(f"\n{metric_name} Detailed Results".center(60))
        print("="*60)
        for field, value in detailed['COMBINED_SEQ'].items():
            if isinstance(value, np.ndarray):
                print(f"{field:.<50}{'[numpy array of shape ' + str(value.shape) + ']':>8}")
            elif isinstance(value, (list, tuple)):
                print(f"{field:.<50}{'[sequence of ' + str(len(value)) + ' values]':>8}")
            elif isinstance(value, (int, float)):
                print(f"{field:.<50}{value:>8.4f}")
            else:
                print(f"{field:.<50}{str(value):>8}")

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            summary_file = os.path.join(output_dir, f"{tracker_name}_summary_results.txt")
            with open(summary_file, "w") as f:
                for metric_name, result in results.items():
                    metric = next(m for m in metric_instances if m.get_name() == metric_name)
                    summary = metric.summary_results({'COMBINED_SEQ': result})
                    
                    f.write("\n" + "="*60 + "\n")
                    f.write(f"{metric_name} Summary Results".center(60) + "\n")
                    f.write("="*60 + "\n")
                    for field, value in summary.items():
                        if isinstance(value, (int, float)):
                            f.write(f"{field:.<50}{value:>8.4f}\n")
                        else:
                            f.write(f"{field:.<50}{str(value):>8}\n")
            
            detailed_file = os.path.join(output_dir, f"{tracker_name}_detailed_results.txt")
            with open(detailed_file, "w") as f:
                for metric_name, result in results.items():
                    metric = next(m for m in metric_instances if m.get_name() == metric_name)
                    detailed = metric.detailed_results({'COMBINED_SEQ': result})
                    
                    f.write("\n" + "="*60 + "\n")
                    f.write(f"{metric_name} Detailed Results".center(60) + "\n") 
                    f.write("="*60 + "\n")
                    for field, value in detailed['COMBINED_SEQ'].items():
                        if isinstance(value, np.ndarray):
                            f.write(f"{field:.<50}{'[numpy array of shape ' + str(value.shape) + ']':>8}\n")
                        elif isinstance(value, (list, tuple)):
                            f.write(f"{field:.<50}{'[sequence of ' + str(len(value)) + ' values]':>8}\n")
                        elif isinstance(value, (int, float)):
                            f.write(f"{field:.<50}{value:>8.4f}\n")
                        else:
                            f.write(f"{field:.<50}{str(value):>8}\n")

    if create_plots and 'HOTA' in results:
        os.makedirs(output_dir, exist_ok=True)
        
        hota_metric = next(m for m in metric_instances if isinstance(m, HOTA))
        hota_metric.plot_single_tracker_results(results['HOTA'], tracker_name, output_dir)
        print(f"Plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate MOT tracker output.")
    parser.add_argument("gt_path", help="Path to the ground truth file.")
    parser.add_argument("tracker_path", help="Path to the tracker output file.")
    parser.add_argument("--metrics", nargs="+", default=['HOTA'], help="List of metrics to compute.")
    parser.add_argument("--output_dir", default="results", help="Output directory to save results.")
    parser.add_argument("--plots", action="store_true", help="Create plots (e.g., HOTA plots).")
    parser.add_argument("--tracker_name", help="Custom name for the tracker (defaults to tracker results filename without extension).")

    args = parser.parse_args()

    evaluate_mot(args.gt_path, args.tracker_path, args.metrics, args.output_dir, args.plots, args.tracker_name)


if __name__ == "__main__":
    main()