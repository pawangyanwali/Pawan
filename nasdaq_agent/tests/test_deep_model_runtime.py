class _FakeTorch:
    def __init__(self):
        self.num_threads = []
        self.interop_threads = []

    def set_num_threads(self, value):
        self.num_threads.append(value)

    def set_num_interop_threads(self, value):
        self.interop_threads.append(value)


def test_configure_torch_runtime_caps_threads(monkeypatch):
    from agent import deep_model

    fake_torch = _FakeTorch()
    monkeypatch.setenv("TORCH_NUM_THREADS", "1")
    monkeypatch.setenv("TORCH_INTEROP_THREADS", "1")
    monkeypatch.setattr(deep_model, "_torch_runtime_configured", False)

    deep_model._configure_torch_runtime(fake_torch)
    deep_model._configure_torch_runtime(fake_torch)

    assert fake_torch.num_threads == [1]
    assert fake_torch.interop_threads == [1]
