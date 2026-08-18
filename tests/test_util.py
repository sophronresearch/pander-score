from pathlib import Path

from pander_score.core import util


def test_load_environment_prefers_nearest_env(tmp_path: Path, monkeypatch) -> None:
    repository = tmp_path / "workspace/repository"
    repository.mkdir(parents=True)
    (tmp_path / ".env").write_text("OUTER=1\n")
    (repository / ".env").write_text("INNER=1\n")
    loaded: list[Path] = []
    monkeypatch.setattr(
        util,
        "load_dotenv",
        lambda path, override=False: loaded.append(Path(path)),
    )

    assert util.load_environment(repository) == repository / ".env"
    assert loaded == [repository / ".env"]
