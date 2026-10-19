"""Tests for logger.progress.fit_verbosity and the set_verbosity / get_verbosity opt-in."""

import logging
import sys

import pytest

import dfm_python
from dfm_python.logger import fit_verbosity, get_logger, get_verbosity, set_verbosity
from dfm_python.utils.errors import ConfigurationError

PKG = "dfm_python"


def _marked(handlers, mark):
    return [h for h in handlers if getattr(h, mark, False)]


@pytest.fixture
def silent_package():
    """Start from the import-time state: only a NullHandler, level NOTSET."""
    pkg = logging.getLogger(PKG)
    for h in list(pkg.handlers):
        pkg.removeHandler(h)
    pkg.addHandler(logging.NullHandler())
    pkg.setLevel(logging.NOTSET)
    pkg.propagate = True
    return pkg


class TestTopLevelExports:
    def test_exported(self):
        assert dfm_python.set_verbosity is set_verbosity
        assert dfm_python.get_verbosity is get_verbosity
        assert "set_verbosity" in dfm_python.__all__
        assert "get_verbosity" in dfm_python.__all__


class TestGetLogger:
    def test_get_logger_attaches_nothing(self, silent_package):
        before = list(silent_package.handlers)
        child = get_logger("dfm_python.some.module")
        assert child.handlers == []
        assert child.propagate is True
        assert silent_package.handlers == before
        assert silent_package.propagate is True
        assert silent_package.level == logging.NOTSET

    def test_silent_by_default(self, silent_package, capsys):
        get_logger("dfm_python.some.module").info("should not appear")
        out = capsys.readouterr()
        assert out.out == "" and out.err == ""


class TestSetVerbosity:
    @pytest.mark.parametrize(
        "level, expected",
        [("debug", logging.DEBUG), ("info", logging.INFO), ("WARNING", logging.WARNING),
         ("error", logging.ERROR), (logging.INFO, logging.INFO), (15, 15)],
    )
    def test_levels(self, silent_package, level, expected):
        set_verbosity(level)
        assert silent_package.level == expected
        assert get_verbosity() == expected

    def test_idempotent_single_handler(self, silent_package):
        for level in ("info", "debug", "info", logging.WARNING):
            set_verbosity(level)
        handlers = _marked(silent_package.handlers, "_dfm_python_verbosity_handler")
        assert len(handlers) == 1
        assert isinstance(handlers[0], logging.StreamHandler)
        assert handlers[0].stream is sys.stdout
        assert len(silent_package.handlers) == 2  # NullHandler + the one stream handler
        assert get_verbosity() == logging.WARNING

    def test_prints_to_stdout(self, silent_package, capsys):
        set_verbosity("info")
        get_logger("dfm_python.some.module").info("hello progress")
        get_logger("dfm_python.some.module").debug("hidden debug")
        out = capsys.readouterr().out
        assert "hello progress" in out
        assert "hidden debug" not in out

    @pytest.mark.parametrize("bad", ["loud", "", None, 1.5, True])
    def test_invalid(self, silent_package, bad):
        with pytest.raises(ConfigurationError, match="Invalid verbosity"):
            set_verbosity(bad)
        assert silent_package.handlers[-1].__class__ is logging.NullHandler

    def test_get_verbosity_default_follows_root(self, silent_package):
        assert get_verbosity() == logging.getLogger().getEffectiveLevel()


class TestFitVerbosity:
    def test_zero_is_noop(self, silent_package, capsys):
        before = list(silent_package.handlers)
        with fit_verbosity(0):
            assert silent_package.handlers == before
            assert silent_package.level == logging.NOTSET
            get_logger("dfm_python.x").info("quiet")
        assert silent_package.handlers == before
        assert capsys.readouterr().out == ""

    @pytest.mark.parametrize("verbose, level", [(1, logging.INFO), (2, logging.DEBUG)])
    def test_attaches_and_cleans_up(self, silent_package, capsys, verbose, level):
        before = list(silent_package.handlers)
        with fit_verbosity(verbose):
            temp = _marked(silent_package.handlers, "_dfm_python_fit_handler")
            assert len(temp) == 1
            assert temp[0].level == level
            assert silent_package.level == level
            get_logger("dfm_python.x").info("info line")
            get_logger("dfm_python.x").debug("debug line")
        assert silent_package.handlers == before
        assert silent_package.level == logging.NOTSET
        out = capsys.readouterr().out
        assert "info line" in out
        assert ("debug line" in out) is (verbose == 2)

    def test_cleans_up_on_exception(self, silent_package):
        before = list(silent_package.handlers)
        with pytest.raises(RuntimeError, match="boom"):
            with fit_verbosity(2):
                raise RuntimeError("boom")
        assert silent_package.handlers == before
        assert silent_package.level == logging.NOTSET

    def test_restores_user_level(self, silent_package):
        silent_package.setLevel(logging.ERROR)
        with fit_verbosity(1):
            assert silent_package.level == logging.INFO
        assert silent_package.level == logging.ERROR

    def test_does_not_reduce_user_verbosity(self, silent_package):
        set_verbosity("debug")
        with fit_verbosity(1):
            assert silent_package.level == logging.DEBUG
        assert silent_package.level == logging.DEBUG

    def test_no_duplicate_handler_with_set_verbosity(self, silent_package, capsys):
        set_verbosity("warning")
        n_before = len(silent_package.handlers)
        with fit_verbosity(1):
            assert len(silent_package.handlers) == n_before
            assert _marked(silent_package.handlers, "_dfm_python_fit_handler") == []
            get_logger("dfm_python.x").info("once")
        assert silent_package.level == logging.WARNING
        assert capsys.readouterr().out.count("once") == 1

    @pytest.mark.parametrize("bad", [-1, 1.0, "1", None, True])
    def test_invalid_verbose(self, silent_package, bad):
        with pytest.raises(ConfigurationError, match="verbose"):
            with fit_verbosity(bad):
                pass
        assert len(silent_package.handlers) == 1


class TestQuietEngineLoggers:
    NAME = "dfm_python.models.fake_engine"

    def _records(self, verbose, level=logging.INFO):
        from dfm_python.logger import quiet_engine_loggers

        pkg = logging.getLogger(PKG)
        records = []

        class _Keep(logging.Handler):
            def emit(self, record):
                records.append(record)

        handler = _Keep(level=logging.DEBUG)
        pkg.addHandler(handler)
        pkg.setLevel(level)
        child = logging.getLogger(self.NAME + ".module")
        try:
            with quiet_engine_loggers(verbose, [self.NAME]):
                child.info("progress")
                child.warning("careful")
        finally:
            pkg.removeHandler(handler)
        return [r.getMessage() for r in records]

    def test_verbose_0_drops_info_but_keeps_warnings(self, silent_package):
        assert self._records(0) == ["careful"]

    def test_verbose_1_changes_nothing(self, silent_package):
        assert self._records(1) == ["progress", "careful"]

    def test_levels_are_restored_also_on_error(self, silent_package):
        from dfm_python.logger import quiet_engine_loggers

        logger = logging.getLogger(self.NAME)
        logger.setLevel(logging.DEBUG)
        with pytest.raises(RuntimeError):
            with quiet_engine_loggers(0, [self.NAME]):
                assert logger.level == logging.WARNING
                raise RuntimeError("boom")
        assert logger.level == logging.DEBUG
        logger.setLevel(logging.NOTSET)

    def test_a_quieter_logger_keeps_its_level(self, silent_package):
        from dfm_python.logger import quiet_engine_loggers

        logger = logging.getLogger(self.NAME)
        logger.setLevel(logging.ERROR)
        with quiet_engine_loggers(0, [self.NAME]):
            assert logger.level == logging.ERROR
        assert logger.level == logging.ERROR
        logger.setLevel(logging.NOTSET)

    def test_invalid_verbose_raises(self):
        from dfm_python.logger import quiet_engine_loggers

        with pytest.raises(ConfigurationError):
            with quiet_engine_loggers(-1, [self.NAME]):
                pass

    def test_level_argument(self, silent_package):
        from dfm_python.logger import quiet_engine_loggers

        logger = logging.getLogger(self.NAME)
        with quiet_engine_loggers(0, [self.NAME], logging.ERROR):
            assert logger.level == logging.ERROR
        assert logger.level == logging.NOTSET
