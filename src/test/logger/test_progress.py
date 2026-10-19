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


# -- engine progress reporting (ported from test_train_logger.py and test_inference_logger.py) --


PROGRESS = "dfm_python.logger.progress"


@pytest.fixture
def progress_records(caplog):
    """Capture records of the progress logger at DEBUG and return (level, message) pairs."""
    caplog.set_level(logging.DEBUG, logger=PROGRESS)

    def records():
        return [(r.levelno, r.getMessage()) for r in caplog.records if r.name == PROGRESS]

    return records


class TestTrainLogger:
    """``iVDFMTrainLogger`` is the old BaseTrainLogger merged with its iVDFM subclass."""

    def test_train_logger_initialization(self):
        from dfm_python.logger.progress import iVDFMTrainLogger

        logger = iVDFMTrainLogger()
        assert logger.model_name == "iVDFM"
        assert logger.verbose is True
        assert logger.iterations == 0 and logger.epochs == 0 and logger.converged is False
        assert logger.metrics_history == [] and logger.start_time is None
        assert iVDFMTrainLogger(verbose=False).verbose is False

    def test_dropped_methods_and_bases(self):
        from dfm_python.logger.progress import iVDFMTrainLogger

        assert iVDFMTrainLogger.__bases__ == (object,)
        assert not hasattr(iVDFMTrainLogger, "log_iteration")

    def test_start_logs_the_config_and_data(self, progress_records):
        from dfm_python.logger.progress import iVDFMTrainLogger

        logger = iVDFMTrainLogger()
        logger.start(config={"max_epochs": 5, "optimizer": "Adam"}, data_info={"data_dim": 3})
        messages = [m for _, m in progress_records()]
        assert messages[:3] == ["=" * 70, "Starting iVDFM training", "=" * 70]
        assert "  max_epochs: 5" in messages and "  optimizer: Adam" in messages and "  data_dim: 3" in messages
        assert logger.start_time is not None

    def test_log_epoch_messages(self, progress_records):
        from dfm_python.logger.progress import iVDFMTrainLogger

        logger = iVDFMTrainLogger()
        logger.log_epoch(epoch=3, elbo=1.5, recon_loss=1.0, kl_loss=0.5, learning_rate=1e-3, grad_norm=2.0)
        quiet = iVDFMTrainLogger(verbose=False)
        quiet.log_epoch(epoch=3, elbo=1.5)
        quiet.log_epoch(epoch=10, elbo=1.25)
        assert progress_records() == [
            (logging.INFO, f"Epoch    3 | ELBO: {1.5:12.6f} | Recon: {1.0:12.6f} | KL: {0.5:12.6f} | LR: 1.000000e-03"
                           " | grad_norm: 2.000000"),
            (logging.INFO, f"Epoch   10 | ELBO: {1.25:12.6f}"),
        ]
        assert logger.epochs == 3 and logger.metrics_history[0]["elbo"] == 1.5

    def test_log_convergence_levels(self, progress_records):
        from dfm_python.logger.progress import iVDFMTrainLogger

        logger = iVDFMTrainLogger()
        logger.log_convergence(converged=True, num_epochs=4, final_loss=0.25, reason="converged")
        logger.log_convergence(converged=False, num_epochs=7, reason="max_epochs")
        assert progress_records() == [
            (logging.INFO, ""),
            (logging.INFO, "✓ Training converged after 4 epochs"),
            (logging.INFO, "  Reason: converged"),
            (logging.INFO, "  Final training loss: 0.250000"),
            (logging.INFO, ""),
            (logging.WARNING, "⚠ Training did not converge after 7 epochs"),
            (logging.INFO, "  Reason: max_epochs"),
        ]
        assert logger.epochs == 7 and logger.converged is False

    def test_end_summarizes(self, progress_records):
        from dfm_python.logger.progress import iVDFMTrainLogger

        logger = iVDFMTrainLogger()
        logger.end()
        assert progress_records() == []  # nothing before start
        logger.start()
        logger.log_epoch(epoch=1, elbo=2.0)
        logger.end(success=False)
        records = progress_records()
        assert (logging.ERROR, "Training failed") in records
        assert (logging.INFO, "  Epochs: 1") in records and (logging.INFO, "  Converged: False") in records


class TestEMProgress:
    """``log_em_iteration`` and ``log_convergence``, called by the DFM EM loop."""

    def test_log_em_iteration(self, progress_records):
        from dfm_python.logger.progress import log_em_iteration

        log_em_iteration(iteration=2, loglik=-10.5, delta=1e-3, max_iter=5, converged="")
        log_em_iteration(iteration=3)
        assert progress_records() == [
            (logging.INFO, f"EM iteration    2 | Log-likelihood: {-10.5:12.6f} | Delta: 1.000000e-03"
                           " | max_iter: 5.000000 | converged: "),
            (logging.INFO, "EM iteration    3"),
        ]

    def test_log_convergence(self, progress_records):
        from dfm_python.logger.progress import log_convergence

        log_convergence(converged=True, num_iter=4, final_loglik=-1.0, reason="converged", model_type="dfm")
        log_convergence(converged=False, num_iter=9, reason="max_iterations_reached", model_type="dfm")
        log_convergence(converged=False, num_epochs=2, final_loss=0.5, model_type="ddfm")
        assert progress_records() == [
            (logging.INFO, "✓ EM algorithm converged after 4 iterations"),
            (logging.INFO, "  Reason: converged"),
            (logging.INFO, "  Final log-likelihood: -1.000000"),
            (logging.WARNING, "⚠ EM algorithm did not converge after 9 iterations"),
            (logging.INFO, "  Reason: max_iterations_reached"),
            (logging.WARNING, "⚠ Training did not converge after 2 epochs"),
            (logging.INFO, "  Final training loss: 0.500000"),
        ]


def test_removed_logger_modules_and_helpers():
    """The old train/inference logger modules (and their DDFM fallbacks) are gone."""
    import importlib

    import dfm_python.logger

    for module in ("train_logger", "inference_logger", "ivdfm_logger", "dfm_logger", "ddfm_logger"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(f"dfm_python.logger.{module}")
    for name in ("BaseTrainLogger", "BaseInferenceLogger", "log_training_start", "log_inference_start"):
        assert not hasattr(dfm_python.logger, name), name
