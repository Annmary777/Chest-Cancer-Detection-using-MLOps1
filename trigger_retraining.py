import json
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# Define constants
INCORRECT_PREDICTIONS_FILE = "incorrect_predictions.json"
INCORRECT_PREDICTIONS_THRESHOLD = 5
RETRAINING_STAGE = "feedback_retraining"  # DVC stage name for retraining

def main():
    # Check if the file exists
    try:
        with open(INCORRECT_PREDICTIONS_FILE, "r") as f:
            incorrect_predictions = json.load(f)
    except FileNotFoundError:
        logging.warning(f"{INCORRECT_PREDICTIONS_FILE} not found. Skipping retraining.")
        return
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding {INCORRECT_PREDICTIONS_FILE}: {e}")
        return

    # Get the number of incorrect predictions
    num_incorrect = len(incorrect_predictions)
    logging.info(f"Number of incorrect predictions: {num_incorrect}")

    # Trigger retraining if the threshold is exceeded
    if num_incorrect > INCORRECT_PREDICTIONS_THRESHOLD:
        logging.info(f"Threshold exceeded: {num_incorrect} > {INCORRECT_PREDICTIONS_THRESHOLD}. Triggering retraining...")
        subprocess.run(["dvc", "repro", RETRAINING_STAGE], check=True)
    else:
        logging.info(f"Threshold not exceeded: {num_incorrect} <= {INCORRECT_PREDICTIONS_THRESHOLD}. Retraining skipped.")

if __name__ == "__main__":
    main()
