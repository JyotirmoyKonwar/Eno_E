from darkguard_trainer.env_client import RemoteEnvClient


def test_client_url_normalization() -> None:
    client = RemoteEnvClient("https://example.com/")
    assert client.base_url.endswith("/")
