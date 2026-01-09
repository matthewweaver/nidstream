"""
Temporal Streaming Simulator

Simulates real-time network flow streaming by replaying temporal test data
with original time delays between flows. Sends predictions to API one by one.

Usage:
    python scripts/stream_temporal.py --speed 1.0  # Real-time
    python scripts/stream_temporal.py --speed 2.0  # 2x speed
    python scripts/stream_temporal.py --speed 0.5  # Half speed
"""

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class TemporalStreamer:
    """Streams network flows with temporal delays."""

    def __init__(self, data_path: str, api_url: str = "http://localhost:8000", speed: float = 1.0, max_flows: int = None):
        """
        Initialize temporal streamer.

        Args:
            data_path: Path to temporal CSV file (must have timestamp column)
            api_url: Base URL of the API
            speed: Speed multiplier (1.0 = real-time, 2.0 = 2x speed)
            max_flows: Maximum number of flows to send (None = all)
        """
        self.data_path = Path(data_path)
        self.api_url = api_url.rstrip("/")
        self.speed = speed
        self.max_flows = max_flows

        self.data = None
        self.stats = {"total_sent": 0, "successful": 0, "failed": 0, "attacks_detected": 0, "start_time": None, "end_time": None}

    def load_data(self):
        """Load temporal test data."""
        logger.info(f"Loading temporal data from {self.data_path}")

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        self.data = pd.read_csv(self.data_path)

        # Validate required columns
        if "timestamp" not in self.data.columns:
            raise ValueError("Data must contain 'timestamp' column")

        # Convert timestamp to datetime
        self.data["timestamp"] = pd.to_datetime(self.data["timestamp"])

        # Sort by timestamp
        self.data = self.data.sort_values("timestamp").reset_index(drop=True)

        # Limit flows if specified
        if self.max_flows:
            self.data = self.data.head(self.max_flows)

        logger.info(f"Loaded {len(self.data)} flows")
        logger.info(f"Time range: {self.data['timestamp'].min()} to {self.data['timestamp'].max()}")

        # Calculate time statistics
        time_diffs = self.data["timestamp"].diff().dropna()
        avg_delay = time_diffs.mean()
        total_duration = self.data["timestamp"].max() - self.data["timestamp"].min()

        logger.info(f"Total duration: {total_duration}")
        logger.info(f"Average delay between flows: {avg_delay}")
        logger.info(f"Stream speed: {self.speed}x")

        estimated_time = total_duration.total_seconds() / self.speed
        logger.info(f"Estimated streaming time: {estimated_time:.1f} seconds ({estimated_time / 60:.1f} minutes)")

    def check_api_health(self) -> bool:
        """Check if API is healthy."""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"API health check failed: {e}")
            return False

    def send_prediction(self, flow_data: dict) -> dict:
        """
        Send a single flow for prediction.

        Args:
            flow_data: Dictionary of flow features

        Returns:
            Prediction result or None if failed
        """
        try:
            # Remove timestamp before sending
            flow_features = {k: v for k, v in flow_data.items() if k != "timestamp"}

            response = requests.post(f"{self.api_url}/predict", json=flow_features, timeout=5)

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Prediction failed: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            logger.error(f"Error sending prediction: {e}")
            return None

    def stream(self):
        """Start streaming flows with temporal delays."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Check API health
        logger.info("Checking API health...")
        if not self.check_api_health():
            raise RuntimeError("API is not healthy. Please start the API first.")

        logger.info("API is healthy. Starting stream...")
        logger.info("=" * 60)

        self.stats["start_time"] = datetime.now()

        # Stream each flow
        for idx, row in self.data.iterrows():
            # Calculate delay from previous flow
            if idx > 0:
                prev_timestamp = self.data.loc[idx - 1, "timestamp"]
                curr_timestamp = row["timestamp"]
                time_diff = (curr_timestamp - prev_timestamp).total_seconds()

                # Apply speed multiplier
                delay = time_diff / self.speed

                if delay > 0:
                    time.sleep(delay)

            # Send prediction
            flow_dict = row.to_dict()
            result = self.send_prediction(flow_dict)

            self.stats["total_sent"] += 1

            if result:
                self.stats["successful"] += 1

                # Track attacks
                if result.get("prediction") == 1 or result.get("prediction_label") == "attack":
                    self.stats["attacks_detected"] += 1
                    logger.warning(
                        f"[{idx + 1}/{len(self.data)}] ATTACK DETECTED | "
                        f"Probability: {result.get('attack_probability', 0):.2%} | "
                        f"Timestamp: {row['timestamp']}"
                    )
                else:
                    logger.info(
                        f"[{idx + 1}/{len(self.data)}] Benign | "
                        f"Probability: {result.get('attack_probability', 0):.2%} | "
                        f"Timestamp: {row['timestamp']}"
                    )
            else:
                self.stats["failed"] += 1

            # Print progress every 10 flows
            if (idx + 1) % 10 == 0:
                progress = (idx + 1) / len(self.data) * 100
                logger.info(f"Progress: {progress:.1f}% ({idx + 1}/{len(self.data)})")

        self.stats["end_time"] = datetime.now()
        self.print_summary()

    def print_summary(self):
        """Print streaming summary statistics."""
        logger.info("=" * 60)
        logger.info("STREAMING SUMMARY")
        logger.info("=" * 60)

        duration = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()

        logger.info(f"Total flows sent: {self.stats['total_sent']}")
        logger.info(f"Successful: {self.stats['successful']}")
        logger.info(f"Failed: {self.stats['failed']}")
        logger.info(f"Attacks detected: {self.stats['attacks_detected']}")

        if self.stats["total_sent"] > 0:
            attack_rate = self.stats["attacks_detected"] / self.stats["total_sent"] * 100
            logger.info(f"Attack rate: {attack_rate:.2f}%")

        logger.info(f"Duration: {duration:.1f} seconds ({duration / 60:.1f} minutes)")

        if duration > 0:
            throughput = self.stats["total_sent"] / duration
            logger.info(f"Throughput: {throughput:.2f} flows/second")

        logger.info("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Temporal network flow streaming simulator")

    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/X_test_temporal.csv",
        help="Path to temporal CSV file (default: data/processed/X_test_temporal.csv)",
    )

    parser.add_argument("--api-url", type=str, default="http://localhost:8000", help="API base URL (default: http://localhost:8000)")

    parser.add_argument("--speed", type=float, default=1.0, help="Speed multiplier (1.0=real-time, 2.0=2x speed, 0.5=half speed)")

    parser.add_argument("--max-flows", type=int, default=None, help="Maximum number of flows to send (default: all)")

    args = parser.parse_args()

    # Create streamer
    streamer = TemporalStreamer(data_path=args.data, api_url=args.api_url, speed=args.speed, max_flows=args.max_flows)

    # Load data
    try:
        streamer.load_data()
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)

    # Start streaming
    try:
        logger.info("Starting temporal streaming...")
        logger.info("Press Ctrl+C to stop")
        streamer.stream()
    except KeyboardInterrupt:
        logger.info("\nStreaming interrupted by user")
        streamer.print_summary()
    except Exception as e:
        logger.error(f"Streaming failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
