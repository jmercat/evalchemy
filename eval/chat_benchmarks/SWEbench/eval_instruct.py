import json
import logging
import tempfile
import os
import sys
from pathlib import Path

from swebench.harness.run_evaluation import main as run_evaluation
from swebench.harness.constants import (
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTION,
)

from patch_cleaner import clean_patch_file

from datasets import load_dataset
from eval.task import BaseBenchmark
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from typing import Any, Dict, Optional

PREDS_PATH = "temp_swebench_preds.json"


class SWEBenchBenchmark(BaseBenchmark):
    """
    SWE-bench (Software Engineering) benchmark for evaluating
    Language Models' ability to resolve GitHub Issues.
    """

    def __init__(
        self,
        dataset_name: str = "princeton-nlp/SWE-bench_Lite",
        debug: bool = False,
        logger: Optional[logging.Logger] = None,
        max_tokens: int = 4096,
        clean_patches: bool = False,
        fix_mode: str = "aggressive",
        system_instruction: Optional[str] = None,
    ):
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.debug = debug
        self.max_tokens = max_tokens
        self.dataset_name = dataset_name
        self.clean_patches = clean_patches  # Flag to control patch cleaning
        self.fix_mode = fix_mode  # Mode for patch cleaning
        """
        Options for <dataset + split(s)>:
        - princeton-nlp/SWE-bench
        - princeton-nlp/SWE-bench_Lite
        - princeton-nlp/SWE-bench_Verified
        """
        self.dataset = load_dataset(self.dataset_name, split="test")

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """
        NOTE: EvalAlchemy's version of SWE-bench evalutes models in a RAG
        based setting (not agentic). The "Oracle" retrieval setting is used for
        this evaluation (to learn more, refer to https://arxiv.org/abs/2310.06770)
        """
        results = {}
        temp_dir_obj = tempfile.TemporaryDirectory()
        temp_dir = temp_dir_obj.name

        instances = [
            x
            for x in load_dataset("princeton-nlp/SWE-bench_oracle", split="test")
            if x["instance_id"] in self.dataset["instance_id"]
        ]

        if self.debug:
            instances = instances[:2]
            self.logger.info("Debug mode: using first 2 instances only")

        all_instances = []
        for idx, instance in enumerate(instances):
            inputs = self._prepare_messages([{"role": "user", "content": instance["text"]}], model)
            all_instances.append(
                Instance(
                    "generate_until",
                    instance,
                    (
                        inputs,
                        {
                            "max_gen_toks": self.max_tokens,
                            "temperature": 0.2,
                            "top_p": 0.95,
                            "do_sample": False,
                        },
                    ),
                    idx,
                )
            )
        self.logger.info("Generating responses for SWE-bench...")
        outputs = self.compute(model, all_instances)

        results = {}
        for idx, (instance, output) in enumerate(zip(instances, outputs)):
            results[instance["instance_id"]] = {
                KEY_INSTANCE_ID: instance["instance_id"],
                KEY_MODEL: model.model_identifier,
                KEY_PREDICTION: output,
            }

        output_file = f"{self.dataset_name.split('/')[-1]}.json"
        output_path = f"{temp_dir}/{output_file}"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        return {"temp_dir_obj": temp_dir_obj, "predictions_path": output_path}

    def clean_prediction_patches(self, predictions_path: str) -> None:
        """
        Clean the patches in the predictions file to improve patch application success.
        
        Args:
            predictions_path: Path to the predictions JSON file
        """
        if not self.clean_patches:
            self.logger.info("Patch cleaning disabled, skipping...")
            return
            
        self.logger.info(f"Cleaning patches in {predictions_path}...")
        patch_counts = {"total": 0, "cleaned": 0, "failed": 0}
        
        try:
            with open(predictions_path, "r") as f:
                predictions = json.load(f)
            
            # Process each instance prediction
            for instance_id, data in predictions.items():
                if KEY_PREDICTION not in data:
                    continue
                    
                prediction = data[KEY_PREDICTION]
                patch_counts["total"] += 1
                
                # Create a temporary file for the patch
                temp_patch_path = os.path.join(os.path.dirname(predictions_path), f"{instance_id}_patch.diff")
                
                # Write the prediction to the temporary file
                with open(temp_patch_path, "w") as f:
                    f.write(prediction)
                
                # Clean the patch
                cleaned_path = clean_patch_file(
                    temp_patch_path, 
                    output_path=temp_patch_path,
                    verbose=True,
                    fix_mode=self.fix_mode
                )
                
                if cleaned_path:
                    # Read the cleaned patch back
                    with open(cleaned_path, "r") as f:
                        cleaned_patch = f.read()
                    
                    # Update the prediction with the cleaned patch
                    data[KEY_PREDICTION] = cleaned_patch
                    patch_counts["cleaned"] += 1
                else:
                    patch_counts["failed"] += 1
                
                # Remove the temporary file
                if os.path.exists(temp_patch_path):
                    os.remove(temp_patch_path)
            
            # Write the updated predictions back to the file
            with open(predictions_path, "w") as f:
                json.dump(predictions, f, indent=2)
            
            self.logger.info(f"Patch cleaning complete: {patch_counts['cleaned']}/{patch_counts['total']} patches cleaned successfully")
            if patch_counts["failed"] > 0:
                self.logger.warning(f"Failed to clean {patch_counts['failed']} patches")
                
        except Exception as e:
            self.logger.error(f"Error cleaning patches: {str(e)}")

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        temp_dir_obj = results["temp_dir_obj"]
        predictions_path = results["predictions_path"]
        
        # Clean patches before evaluation
        # self.clean_prediction_patches(predictions_path)

        report_path = run_evaluation(
            dataset_name=self.dataset_name,
            split="test",
            predictions_path=predictions_path,
            instance_ids=None,
            max_workers=32,
            force_rebuild=False,
            cache_level="none",
            clean=False,
            open_file_limit=4096,
            run_id="swe-bench-evalchemy",
            timeout=1800,
            namespace="swebench",
            rewrite_reports=False,
            modal=False,
            instance_image_tag="v1",
            report_dir=".",
        )

        temp_dir_obj.cleanup()
        if report_path is None:
            self.logger.error("Error evaluating SWE-bench")
            return None

        return json.load(open(report_path))

    def run_benchmark(self, model: LM) -> Dict[str, float]:
        self.logger.info("Starting SWE-bench evaluation")
        try:
            results = self.generate_responses(model)
            return self.evaluate_responses(results)
        except Exception as e:
            self.logger.error(f"Error running benchmark: {str(e)}")
            return {"error": str(e)}
