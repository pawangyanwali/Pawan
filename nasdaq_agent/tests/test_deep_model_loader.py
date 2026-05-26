import pytest
pytestmark = pytest.mark.slow

import sys
import time
import types
from concurrent.futures import ThreadPoolExecutor


def test_cluster_model_load_is_single_flight(monkeypatch, tmp_path):
    import agent.deep_model as dm

    model_path = tmp_path / "cluster_a.pt"
    scaler_path = tmp_path / "cluster_a.pkl"
    model_path.write_bytes(b"weights")

    build_count = {"value": 0}
    load_count = {"value": 0}

    class FakeModel:
        def load_state_dict(self, state):
            self.state = state

        def eval(self):
            self.eval_called = True

    def fake_build_model(n_tickers):
        build_count["value"] += 1
        time.sleep(0.01)
        return FakeModel()

    fake_torch = types.ModuleType("torch")

    def fake_load(*args, **kwargs):
        load_count["value"] += 1
        return {"ok": True}

    fake_torch.load = fake_load

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(dm, "_build_model", fake_build_model)
    monkeypatch.setattr(dm, "_load_scaler", lambda path: "scaler")
    monkeypatch.setattr(dm, "_cluster_models", {"A": None, "B": None, "C": None})
    monkeypatch.setattr(dm, "_cluster_scalers", {"A": None, "B": None, "C": None})
    monkeypatch.setattr(dm, "_cluster_trained", {"A": False, "B": False, "C": False})
    monkeypatch.setattr(
        dm,
        "_CLUSTER_CONFIGS",
        {"A": {"path": model_path, "scaler": scaler_path}},
    )

    with ThreadPoolExecutor(max_workers=8) as executor:
        models = list(executor.map(lambda _: dm._get_cluster_model("A"), range(16)))

    assert len({id(model) for model in models}) == 1
    assert build_count["value"] == 1
    assert load_count["value"] == 1
    assert dm._cluster_scalers["A"] == "scaler"
    assert dm._cluster_trained["A"] is True
