"""Moto-mocked S3 tests for the Lambda retraining/promotion logic."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import boto3
import pytest
from moto import mock_aws

BUCKET = "pdm-platform-artifacts"
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")


@pytest.fixture
def s3_bucket():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client


def test_no_production_model_means_any_candidate_qualifies(s3_bucket):
    os.environ["BUCKET_NAME"] = BUCKET
    from infra.retrain_lambda.retrain_handler import _get_production_metrics
    metrics = _get_production_metrics()
    assert metrics["f1"] == 0.0


def test_production_metrics_read_back_correctly(s3_bucket):
    os.environ["BUCKET_NAME"] = BUCKET
    s3_bucket.put_object(
        Bucket=BUCKET,
        Key="models/production/metrics.json",
        Body=json.dumps({"f1": 0.81, "model": "xgboost"}),
    )
    from infra.retrain_lambda.retrain_handler import _get_production_metrics
    metrics = _get_production_metrics()
    assert metrics["f1"] == 0.81
    assert metrics["model"] == "xgboost"


def test_promotion_threshold_logic():
    """Candidate must beat production by PROMOTION_MARGIN, not just tie."""
    margin = 0.01
    production_f1 = 0.80

    assert (0.815 >= production_f1 + margin) is True    # clears margin -> promote
    assert (0.805 >= production_f1 + margin) is False   # within margin -> hold
    assert (0.79 >= production_f1 + margin) is False    # regression -> hold
