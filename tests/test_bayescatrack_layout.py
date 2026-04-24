import bayescatrack
import track2p_pyrecest_bridge
import track2p_pyrecest_bridge.reference as bridge_reference
import track2p_pyrecest_bridge.registration as bridge_registration
import track2p_pyrecest_bridge.track2p_registration as bridge_track2p_registration
from bayescatrack import association
from bayescatrack import io as bayescatrack_io
from bayescatrack import reference, registration, track2p_registration
from bayescatrack.datasets import track2p as bayescatrack_track2p
from tests._support import assert_module_reexports, run_module


def test_root_package_reexports_expected_public_api():
    assert_module_reexports(bayescatrack, track2p_pyrecest_bridge)


def test_subpackages_expose_expected_wrappers():
    for module in (association, bayescatrack_track2p, bayescatrack_io):
        assert_module_reexports(module, track2p_pyrecest_bridge)


def test_recent_bridge_modules_are_available_under_bayescatrack():
    for module, source_module in (
        (reference, bridge_reference),
        (registration, bridge_registration),
        (track2p_registration, bridge_track2p_registration),
    ):
        assert_module_reexports(module, source_module)


def test_bayescatrack_module_entry_point_help():
    proc = run_module("-m", "bayescatrack", "--help")
    assert "summary" in proc.stdout
    assert "export" in proc.stdout
