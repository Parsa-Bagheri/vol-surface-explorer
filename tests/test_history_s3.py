from io import BytesIO
from pathlib import Path
import pytest

from src.history_store import LocalObservationStore, ObservationError
from src.history_demo import seed_demo
from src.history_s3 import export_observation, verify_export, S3ObservationArchive


class FakePreconditionError(Exception):
    response = {"ResponseMetadata": {"HTTPStatusCode": 412}}


class FakeS3:
    def __init__(self, fail_after=None):
        self.objects = {}
        self.order = []
        self.fail_after = fail_after

    def put_object(self, Bucket, Key, Body, **kwargs):
        assert kwargs["IfNoneMatch"] == "*"
        if self.fail_after is not None and len(self.order) >= self.fail_after:
            raise OSError("Simulated S3 interruption")
        if (Bucket, Key) in self.objects:
            raise FakePreconditionError()
        self.objects[(Bucket, Key)] = Body
        self.order.append(Key)

    def get_object(self, Bucket, Key):
        return {"Body": BytesIO(self.objects[(Bucket, Key)])}


def test_verified_export_and_s3_transport_round_trip_with_commit_last(tmp_path):
    store = LocalObservationStore(tmp_path / "lake")
    seed_demo(store)
    output = tmp_path / "export"
    exported = export_observation(store, "synthetic-a", output)
    assert verify_export(output)["bundle_sha256"] == exported["bundle_sha256"]
    client = FakeS3()
    archive = S3ObservationArchive(client, "fake-test-bucket")
    uploaded = archive.upload(output)
    assert client.order[-1].endswith("commit.json")
    assert archive.upload(output) == uploaded  # Conditional immutable retry.
    downloaded = tmp_path / "downloaded"
    archive.download(uploaded["commit_key"], downloaded)
    restored = LocalObservationStore(downloaded)
    assert restored.replay("synthetic-a")["matched"]
    with pytest.raises(ObservationError, match="already exists"):
        export_observation(store, "synthetic-a", output)


def test_interrupted_s3_upload_never_publishes_commit(tmp_path):
    store = LocalObservationStore(tmp_path / "lake")
    seed_demo(store)
    output = tmp_path / "export"
    export_observation(store, "synthetic-b", output)
    client = FakeS3(fail_after=2)
    with pytest.raises(OSError, match="interruption"):
        S3ObservationArchive(client, "fake-test-bucket").upload(output)
    assert not any(key.endswith("commit.json") for key in client.order)
    assert store.load("synthetic-b")[0]["partial"]


def test_athena_projection_is_typed_bounded_and_refuses_version_overwrite(tmp_path):
    import pyarrow.parquet as pq
    from scripts.export_athena_projection import export_projection
    store = LocalObservationStore(tmp_path / "lake")
    seed_demo(store)
    artifact = store.process("synthetic-a")
    root = tmp_path / "analytics"
    exported = export_projection(store, "synthetic-a", artifact["processing_id"], root)
    schema = pq.read_schema(exported["output"])
    assert "ticker" not in schema.names
    assert schema.field("valuation_time_utc").type.unit == "us"
    assert not exported["aws_executed"]
    with pytest.raises(ObservationError, match="already exists"):
        export_projection(store, "synthetic-a", artifact["processing_id"], root)
