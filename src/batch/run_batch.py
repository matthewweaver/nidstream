"""
Batch Processing Pipeline

Process network flow logs in batches and save predictions.
Useful for:
- Processing historical logs
- Daily/weekly batch scoring
- Large-scale analysis
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from inference_pipeline.predict import load_detector

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def process_batch(input_file: str, output_file: str, model_dir: str = "models", chunk_size: int = None, save_summary: bool = True):
    """
    Process a batch of network flows and save predictions.

    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        model_dir: Directory containing model files
        chunk_size: Process in chunks (for large files)
        save_summary: Save summary statistics
    """

    # Load model
    logger.info(f"Loading model from {model_dir}")
    detector = load_detector(model_dir)
    model_info = detector.get_model_info()
    logger.info(f"Loaded model: {model_info.get('model_name', 'Unknown')}")

    # Process data
    if chunk_size:
        logger.info(f"Processing in chunks of {chunk_size}")
        process_in_chunks(input_file, output_file, detector, chunk_size)
    else:
        logger.info(f"Loading data from {input_file}")
        data = pd.read_csv(input_file)
        logger.info(f"Loaded {len(data)} flows")

        # Make predictions
        logger.info("Making predictions...")
        results = detector.predict_batch(data)

        # Save results
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")

    # Generate summary
    if save_summary:
        generate_summary(output_file, model_info)


def process_in_chunks(input_file: str, output_file: str, detector, chunk_size: int):
    """Process large files in chunks to manage memory."""

    first_chunk = True
    total_processed = 0
    total_attacks = 0

    for chunk in pd.read_csv(input_file, chunksize=chunk_size):
        logger.info(f"Processing chunk of {len(chunk)} flows...")

        # Make predictions
        results = detector.predict_batch(chunk)

        # Save (append mode after first chunk)
        mode = "w" if first_chunk else "a"
        header = first_chunk
        results.to_csv(output_file, mode=mode, header=header, index=False)

        # Update stats
        total_processed += len(results)
        total_attacks += (results["prediction"] == 1).sum()
        first_chunk = False

        logger.info(f"Processed {total_processed} flows so far...")

    logger.info(f"Completed: {total_processed} flows, {total_attacks} attacks detected")


def generate_summary(predictions_file: str, model_info: dict):
    """Generate and save summary statistics."""

    logger.info("Generating summary...")

    # Load predictions
    results = pd.read_csv(predictions_file)

    # Calculate statistics
    total_flows = len(results)
    n_attacks = (results["prediction"] == 1).sum()
    n_benign = (results["prediction"] == 0).sum()
    attack_rate = n_attacks / total_flows if total_flows > 0 else 0

    # High confidence attacks
    high_conf_attacks = results[(results["prediction"] == 1) & (results["attack_probability"] >= 0.8)]

    # Create summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "input_file": str(predictions_file),
        "model_name": model_info.get("model_name", "Unknown"),
        "model_type": model_info.get("model_type", "Unknown"),
        "total_flows": int(total_flows),
        "attacks_detected": int(n_attacks),
        "benign_flows": int(n_benign),
        "attack_rate_percent": round(attack_rate * 100, 2),
        "high_confidence_attacks": len(high_conf_attacks),
        "avg_attack_probability": round(results[results["prediction"] == 1]["attack_probability"].mean(), 4) if n_attacks > 0 else 0,
        "max_attack_probability": round(results["attack_probability"].max(), 4),
    }

    # Save summary
    summary_file = str(Path(predictions_file).parent / f"{Path(predictions_file).stem}_summary.json")
    import json

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Summary saved to {summary_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 80)
    print(f"Total Flows Processed: {total_flows:,}")
    print(f"Attacks Detected:      {n_attacks:,} ({attack_rate * 100:.2f}%)")
    print(f"Benign Flows:          {n_benign:,}")
    print(f"High Confidence Attacks: {len(high_conf_attacks):,}")
    print(f"Average Attack Prob:   {summary['avg_attack_probability']:.4f}")
    print("=" * 80 + "\n")

    return summary


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Batch processing for network intrusion detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python run_batch.py --input data/flows.csv --output predictions.csv
  
  # Process large file in chunks
  python run_batch.py --input large_file.csv --output pred.csv --chunk-size 10000
  
  # Specify model directory
  python run_batch.py --input flows.csv --output pred.csv --model-dir ./models
        """,
    )

    parser.add_argument("--input", "-i", required=True, help="Input CSV file with network flows")
    parser.add_argument("--output", "-o", required=True, help="Output CSV file for predictions")
    parser.add_argument("--model-dir", "-m", default="models", help="Directory containing model files (default: models)")
    parser.add_argument("--chunk-size", "-c", type=int, default=None, help="Process in chunks (for large files)")
    parser.add_argument("--no-summary", action="store_true", help="Do not generate summary file")

    args = parser.parse_args()

    try:
        process_batch(
            input_file=args.input,
            output_file=args.output,
            model_dir=args.model_dir,
            chunk_size=args.chunk_size,
            save_summary=not args.no_summary,
        )

        logger.info("Batch processing completed successfully!")

    except Exception as e:
        logger.error(f"Batch processing failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
